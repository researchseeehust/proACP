"""Contact-target interface of "Humanoid Mobility via Attention-based Contacts" (He, Wen, Zhang et al., arXiv:2610.31608)
for the G1 on the PHP box terrain: the tracker's command, contact geometry and the Stage 1 reward.

The command is the tracker's input other than proprioception:

- for each of the 9 :data:`proacp.robot.CONTACT_BODIES`, the terrain point of its next new contact within
  ``horizon_s`` (``target_point="terrain"``; ``"link"``: the reference body link at the onset frame), zeros when there
  is none;
- the goal command: the reference root ``goal_s`` ahead (one draw per episode) as the paper's bounded direction
  d_xy / max(|d_xy|, ``goal_radius``) and relative heading (goal reaching: the terrain's goal, radius 0.5 m);
- the paper's 9 x 7 local height scan (0.8 x 0.6 m at 0.1 m: root height above the terrain), unless ``blind``;

positions in the robot's base frame (root position, yaw-only rotation). Contact labels come from
:mod:`proacp.clips.labels` (one set per terrain variant) and are debounced here.

Touching is a soft (sigmoid) function of the body's signed distance to the episode's boxes and the ground, on a sphere
approximation of its collision geoms (:class:`BodySpheres`), so reward terms on it have a gradient through the world
model; the paper measures contact forces, which the world model does not predict.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from proacp.robot import CONTACT_BODIES, NB, TRACKED_BODIES, action_scale, JOINT_NAMES, get_spec, pd_params
from proacp.tasks.tracking import MDP
from proacp.clips.library import CONTROL_DT
from proacp.sim.terrain import box_closest_point, box_sdf, grid_offsets, height_scan
from proacp.sim.rotations import rot_angle_sq


@dataclass
class ContactCfg:
    contacts_dir: str | None = None
    """proacp.clips.labels output; default: the motion directory + ``_contacts``."""
    horizon_s: float = 1.5
    goal_s: tuple[float, float] = (0.5, 1.5)
    goal_radius: float = 0.3
    """Goal direction d_xy / max(|d_xy|, goal_radius) (the paper's reference-following value)."""
    scan_x: tuple[float, float, int] = (-0.4, 0.4, 9)
    scan_y: tuple[float, float, int] = (-0.3, 0.3, 7)
    """Local scan grid of the command (not ``blind``) in the base frame: the paper's 0.8 x 0.6 m at 0.1 m."""
    target_noise: float = 0.05
    scan_noise: float = 0.05
    """Uniform noise (m) on the target positions and the scan, every step (paper, G1: 0.10 and 0.10)."""
    debounce: int = 2
    """Contact gaps, then contacts, of at most this many frames are removed from the labels."""
    target_point: str = "terrain"
    """Target position: "terrain" (the terrain point the body touches, as the contact planner outputs) or "link" (the
    body's target frame link)."""
    target_min_steps: int = 10
    """Targets come only from contacts of at least this many frames (the planner's hindsight labels: 10)."""
    w_label: float = 1.0
    touch_dist: float = 0.01
    touch_temp: float = 0.005
    """Soft touch = sigmoid((touch_dist - signed distance) / touch_temp)."""
    near_boxes: int = 4
    """Boxes per sample the touch test considers (nearest to the root)."""
    blind: bool = False
    """No local scan in the command (not the paper's): the tracker sees only the contact targets and the goal, the
    terrain reaches it through the contact planner only. The world model and the critic keep their height scan."""


def _runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and ends (exclusive) of the runs of True in the 1-D bool array ``x``."""
    e = np.diff(np.r_[0, x.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def debounce(c: np.ndarray, n: int) -> np.ndarray:
    """(T, B) bool labels with interior gaps of at most n frames filled, then contacts of at most n frames removed."""
    c = c.copy()
    T = len(c)
    for b in range(c.shape[1]):
        x = c[:, b]
        for s, e in zip(*_runs(~x)):
            if s > 0 and e < T and e - s <= n:
                x[s:e] = True
        for s, e in zip(*_runs(x)):
            if e - s <= n:
                x[s:e] = False
    return c


def drop_short(c: np.ndarray, n: int) -> np.ndarray:
    """(T, B) bool labels without the contacts shorter than n frames."""
    c = c.copy()
    for b in range(c.shape[1]):
        for s, e in zip(*_runs(c[:, b])):
            if e - s < n:
                c[s:e, b] = False
    return c


def next_onsets(c: np.ndarray, horizon: int) -> np.ndarray:
    """(T, B) frame of each body's next contact onset after t and within ``horizon`` frames, -1: none."""
    T = len(c)
    out = np.full(c.shape, -1, dtype=np.int64)
    t = np.arange(T)
    for b in range(c.shape[1]):
        on = np.flatnonzero(c[1:, b] & ~c[:-1, b]) + 1
        if len(on) == 0:
            continue
        k = np.searchsorted(on, t, side="right")
        nxt = on[np.minimum(k, len(on) - 1)]
        out[:, b] = np.where((k < len(on)) & (nxt - t <= horizon), nxt, -1)
    return out


MESH_POINTS = 24
MESH_RADIUS = 0.01
"""A convex mesh hull: this many of its vertices (farthest-point sampled), as spheres of this radius just inside it."""


def _farthest_points(x: np.ndarray, k: int) -> np.ndarray:
    idx = [int(np.argmax(np.linalg.norm(x - x.mean(0), axis=-1)))]
    d = np.linalg.norm(x - x[idx[0]], axis=-1)
    for _ in range(min(k, len(x)) - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(x - x[idx[-1]], axis=-1))
    return x[idx]


class BodySpheres:
    """Spheres covering each contact body's collision geoms, in its target link's frame: spheres exactly, capsules as
    3 spheres along their axis, convex meshes (the G1's limb hulls) as :data:`MESH_POINTS` spheres on their surface."""

    def __init__(self):
        m = get_spec().compile()
        pts, rad, owner = [], [], []
        for b, (frame, links) in enumerate(CONTACT_BODIES.values()):
            fid = m.body(frame).id
            for link in links:
                lid = m.body(link).id
                if lid == fid:
                    p0, R0 = np.zeros(3), np.eye(3)
                else:  # a jointless child of the frame link
                    assert m.body_parentid[lid] == fid and m.body_jntnum[lid] == 0, link
                    p0, R0 = m.body_pos[lid].copy(), _quat_mat(m.body_quat[lid])
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] != lid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                        continue
                    c, Rg = p0 + R0 @ m.geom_pos[g], R0 @ _quat_mat(m.geom_quat[g])
                    t, size = m.geom_type[g], m.geom_size[g]
                    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                        pts.append(c), rad.append(size[0]), owner.append(b)
                    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        for u in (-1.0, 0.0, 1.0):
                            pts.append(c + u * size[1] * Rg[:, 2]), rad.append(size[0]), owner.append(b)
                    elif t == mujoco.mjtGeom.mjGEOM_MESH:
                        i = m.geom_dataid[g]
                        v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(np.float64)
                        sel = _farthest_points(v, MESH_POINTS)
                        d = sel - v.mean(0)
                        sel = sel - MESH_RADIUS * d / np.linalg.norm(d, axis=-1, keepdims=True).clip(min=1e-6)
                        for x in sel:
                            pts.append(c + Rg @ x), rad.append(MESH_RADIUS), owner.append(b)
                    else:
                        raise ValueError(f"{m.geom(g).name}: unsupported geom type {t}")
        order = np.argsort(owner, kind="stable")
        self.pts = torch.tensor(np.array(pts)[order], dtype=torch.float32)
        self.rad = torch.tensor(np.array(rad)[order], dtype=torch.float32)
        owner = np.array(owner)[order]
        self.owner = torch.tensor(owner)
        self.bounds = [(int(np.searchsorted(owner, b)), int(np.searchsorted(owner, b, side="right"))) for b in range(NB)]


def _quat_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def to_base(x: torch.Tensor, p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """World points x (B, ..., 3) in the base frame of root (p, R): root position, yaw-only rotation."""
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    lead = (1,) * (x.dim() - 2)
    cs, sn = torch.cos(yaw).view(-1, *lead), torch.sin(yaw).view(-1, *lead)
    d = x - p.view(p.shape[0], *lead, 3)
    return torch.stack((cs * d[..., 0] + sn * d[..., 1], cs * d[..., 1] - sn * d[..., 0], d[..., 2]), -1)


def from_base(x: torch.Tensor, p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_base`: base-frame points x (B, ..., 3) of root (p, R) in the world."""
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    lead = (1,) * (x.dim() - 2)
    cs, sn = torch.cos(yaw).view(-1, *lead), torch.sin(yaw).view(-1, *lead)
    w = torch.stack((cs * x[..., 0] - sn * x[..., 1], sn * x[..., 0] + cs * x[..., 1], x[..., 2]), -1)
    return w + p.view(p.shape[0], *lead, 3)


def goal_command(S, goal_p: torch.Tensor, goal_yaw: torch.Tensor, radius: float) -> torch.Tensor:
    """(B, 3) the paper's goal command of state S toward world point ``goal_p`` (B, 3) with heading ``goal_yaw`` (B,):
    the bounded base-frame direction d_xy / max(|d_xy|, radius) and the relative heading wrapped to [-pi, pi)."""
    p, R = S[0], S[1]
    d = to_base(goal_p, p, R)[..., :2]
    d = d / torch.clamp(d.norm(dim=-1, keepdim=True), min=radius)
    return torch.cat((d, wrap(goal_yaw - torch.atan2(R[..., 1, 0], R[..., 0, 0])).unsqueeze(-1)), -1)


PLACE_WINDOW = 5  # control steps of commands a new contact is matched against


def placement_error(cp: torch.Tensor, tw_win: torch.Tensor, tv_win: torch.Tensor) -> torch.Tensor:
    """(B, NB) distance from each body's contact point cp (B, NB, 3) to the nearest valid target of the last steps'
    commands (tw_win (L, B, NB, 3), tv_win (L, B, NB)); inf where none was valid. A contact landing a step or two after
    its reference onset already sees the next target, so the target of the onset step alone over-counts by a stride."""
    d = (cp.unsqueeze(0) - tw_win).norm(dim=-1)
    return torch.where(tv_win, d, torch.inf).min(0).values


class ContactTask:
    """Commands, contact geometry, reward terms and critic input of the contact interface (simulator and imagination).
    Per-frame arrays are indexed by (reference frame f, terrain variant v); a terrain index v >= V (a goal-reaching
    terrain, :mod:`proacp.tasks.goal`) reads variant 0 there, which only reference-free code then ignores."""

    def __init__(self, cfg: ContactCfg, mdp: MDP, motion_dir: str, device: str):
        self.cfg, self.mdp, self.motion = cfg, mdp, mdp.motion
        m, dev = self.motion, device
        cdir = cfg.contacts_dir or motion_dir.rstrip("/") + "_contacts"
        H = int(round(cfg.horizon_s / CONTROL_DT))
        V = m.num_variants
        labels, onset = np.zeros((m.num_frames, V, NB), bool), np.full((m.num_frames, V, NB), -1, np.int64)
        touched = np.zeros((m.num_frames, V, NB, 3), np.float32)
        for c, name in enumerate(m.names):
            path = os.path.join(cdir, name + ".npz")
            assert os.path.exists(path), f"{path} missing: run python -m proacp.clips.labels"
            d = np.load(path)
            assert list(d["body_names"]) == list(CONTACT_BODIES), d["body_names"]
            raw = d["contact_info"][..., 0] > 0.5  # (V_clip, T, NB)
            s, n = int(m.clip_start[c]), int(m.clip_len[c])
            assert raw.shape[1] == n, (name, n, raw.shape)
            for v in range(V):
                vc = v if v < len(raw) else 0  # a clip with fewer variants repeats its first (terrain.load_clip_boxes)
                lab = debounce(raw[vc], cfg.debounce)
                on = next_onsets(drop_short(lab, cfg.target_min_steps) if cfg.target_min_steps else lab, H)
                touched[s : s + n, v] = d["contact_pos_w"][vc]
                labels[s : s + n, v], onset[s : s + n, v] = lab, np.where(on >= 0, on + s, -1)
        self.V = V
        self.labels = torch.tensor(labels, device=dev)
        self.next_valid = torch.tensor(onset >= 0, device=dev)
        self.next_onset = torch.tensor(onset, device=dev)
        self.link_idx = torch.tensor([TRACKED_BODIES.index(frame) for frame, _ in CONTACT_BODIES.values()], device=dev)
        on = torch.tensor(np.maximum(onset, 0), device=dev)  # (F, V, NB)
        if cfg.target_point == "terrain":  # the terrain point touched at the onset frame (NaN only away from contact)
            vv = torch.arange(V, device=dev).view(1, V, 1)
            touched = torch.tensor(touched, device=dev)[on, vv, torch.arange(NB, device=dev).view(1, 1, NB)]
            assert not touched[self.next_valid].isnan().any(), "a contact onset without a touched point"
            self.next_pos = torch.nan_to_num(touched)  # (F, V, NB, 3)
        else:
            assert cfg.target_point == "link", cfg.target_point
            self.next_pos = m.body_p[on, self.link_idx.view(1, 1, NB)]  # reference links at the onset frame
        sp = BodySpheres()
        self.sph_pts, self.sph_rad, self.sph_owner = sp.pts.to(dev), sp.rad.to(dev), sp.owner.to(dev)
        self.sph_bounds = sp.bounds
        self.scan_offsets = grid_offsets(cfg.scan_x, cfg.scan_y, device=dev)
        self.goal_steps = (int(round(cfg.goal_s[0] / CONTROL_DT)), int(round(cfg.goal_s[1] / CONTROL_DT)))
        self.cmd_dim = NB * 3 + 3 + (0 if cfg.blind else self.scan_offsets.shape[0])

    # ---------------- commands ----------------
    def sample_goal_steps(self, n: int) -> torch.Tensor:
        lo, hi = self.goal_steps
        return torch.randint(lo, hi + 1, (n,), device=self.labels.device)

    def rv(self, v: torch.Tensor) -> torch.Tensor:
        """Terrain index -> the clip variant the per-frame label arrays hold (0 for goal-reaching terrains)."""
        return torch.where(v < self.V, v, 0)

    def command(self, S, f: torch.Tensor, v: torch.Tensor, goal_steps: torch.Tensor | None, noise: bool, targets_w: torch.Tensor | None = None,
                valid: torch.Tensor | None = None, goal: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
        """(B, cmd_dim): next-contact targets, goal command (and local scan) of state S on reference frame f, terrain v.
        ``targets_w`` (B, 9, 3) world and ``valid`` (B, 9): targets from another source (the contact planner); ``goal``
        (B, 3): the goal command of goal reaching instead of the reference's."""
        c = self.cfg
        p, R = S[0], S[1]
        tgt = to_base(self.next_pos[f, self.rv(v)] if targets_w is None else targets_w, p, R)
        if noise:
            tgt = tgt + (2 * torch.rand_like(tgt) - 1) * c.target_noise
        valid = self.next_valid[f, self.rv(v)] if valid is None else valid
        tgt = torch.where(valid.unsqueeze(-1), tgt, 0.0)
        if goal is None:
            goal = self.reference_goal(S, f, goal_steps)
        if c.blind:
            return torch.cat((tgt.flatten(-2), goal), -1)
        scan = height_scan(p, R, self.motion.boxes(f, v), self.scan_offsets)
        if noise:
            scan = scan + (2 * torch.rand_like(scan) - 1) * c.scan_noise
        return torch.cat((tgt.flatten(-2), goal, scan), -1)

    def reference_goal(self, S, f: torch.Tensor, goal_steps: torch.Tensor, radius: float | None = None) -> torch.Tensor:
        """(B, 3) the goal command of the reference root ``goal_steps`` ahead (capped at the clip's end)."""
        m = self.motion
        fg = torch.minimum(f + goal_steps, m.clip_end[m.clip_id[f]])
        yaw_g = torch.atan2(m.root_R[fg][..., 1, 0], m.root_R[fg][..., 0, 0])
        return goal_command(S, m.root_p[fg], yaw_g, self.cfg.goal_radius if radius is None else radius)

    # ---------------- contact geometry ----------------
    def _sphere_centers(self, fk) -> torch.Tensor:
        bp, bR = fk[:2]
        lp, lR = bp[:, self.link_idx][:, self.sph_owner], bR[:, self.link_idx][:, self.sph_owner]
        return lp + (lR @ self.sph_pts.unsqueeze(-1)).squeeze(-1)  # (B, P, 3)

    def body_distance(self, S, boxes: torch.Tensor, fk=None) -> torch.Tensor:
        """(B, 9) signed distance of each contact body's spheres to the ground and the nearest of ``boxes`` (B, K, 8);
        ``fk``: ``kin.forward(*S)`` if already computed."""
        x = self._sphere_centers(self.mdp.kin.forward(*S) if fk is None else fk)
        near = self._near_boxes(S[0], boxes)
        d = x[..., 2]
        for k in range(near.shape[1]):
            d = torch.minimum(d, box_sdf(x, near[:, k]))
        d = d - self.sph_rad
        return torch.stack([d[:, s:e].min(-1).values for s, e in self.sph_bounds], -1)

    @torch.no_grad()
    def contact_point(self, S, boxes: torch.Tensor, fk=None) -> torch.Tensor:
        """(B, 9, 3) the terrain point (ground or the nearest boxes) closest to each contact body's sphere
        approximation: where it touches when in contact."""
        x = self._sphere_centers(self.mdp.kin.forward(*S) if fk is None else fk)
        near = self._near_boxes(S[0], boxes)
        d, pt = x[..., 2], torch.cat((x[..., :2], torch.zeros_like(x[..., 2:])), -1)
        for k in range(near.shape[1]):
            dk = box_sdf(x, near[:, k])
            closer = dk < d
            d, pt = torch.where(closer, dk, d), torch.where(closer.unsqueeze(-1), box_closest_point(x, near[:, k]), pt)
        d = d - self.sph_rad
        out = []
        for s, e in self.sph_bounds:
            i = d[:, s:e].argmin(-1)
            out.append(pt[:, s:e].gather(1, i.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1))
        return torch.stack(out, 1)

    def nearest_body_point(self, fk, target: torch.Tensor) -> torch.Tensor:
        """(B, 9, 3) the surface point of each contact body's sphere approximation nearest to its ``target`` (B, 9, 3)
        world; ``fk`` = ``kin.forward(*S)``."""
        x = self._sphere_centers(fk)
        d = target[:, self.sph_owner] - x
        dist = torch.sqrt((d**2).sum(-1) + 1e-9)
        surf = x + (self.sph_rad / dist).unsqueeze(-1) * d
        gap = dist - self.sph_rad
        out = [surf[:, s:e].gather(1, gap[:, s:e].argmin(-1).view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1) for s, e in self.sph_bounds]
        return torch.stack(out, 1)

    @torch.no_grad()
    def _near_boxes(self, p: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """The ``near_boxes`` boxes (of (B, K, 8)) nearest to the root in xy, padding last."""
        k = min(self.cfg.near_boxes, boxes.shape[1])
        if k == boxes.shape[1]:
            return boxes
        cx, cy, _, c, s, hx, hy, _ = boxes.unbind(-1)
        dx, dy = p[:, 0:1] - cx, p[:, 1:2] - cy
        q = torch.stack(((c * dx + s * dy).abs() - hx, (c * dy - s * dx).abs() - hy), -1)
        dist = torch.where(hx > 0, torch.relu(q).norm(dim=-1), math.inf)
        idx = dist.topk(k, -1, largest=False).indices
        return boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]))

    def touch(self, S, boxes: torch.Tensor, fk=None) -> torch.Tensor:
        """(B, 9) soft touch in [0, 1]."""
        c = self.cfg
        return torch.sigmoid((c.touch_dist - self.body_distance(S, boxes, fk)) / c.touch_temp)

    def label_reward(self, S, f: torch.Tensor, v: torch.Tensor, boxes: torch.Tensor, fk=None) -> torch.Tensor:
        """Asymmetric contact-label term in [-1, 1]: touching bodies score +1 if the reference touches, else -1."""
        return (self.touch(S, boxes, fk) * (2 * self.labels[f, self.rv(v)].float() - 1)).mean(-1)

    def critic_input(self, S, f, v, goal_steps, xi_n) -> torch.Tensor:
        """Privileged tracking features (reference errors, sim context, height scan) and the noise-free command."""
        boxes = self.motion.boxes(f, v)
        return torch.cat((self.mdp.critic_features(S, f, xi_n, boxes), self.command(S, f, v, goal_steps, noise=False)), -1)


# ---------------- the contact paper's Stage 1 reward ----------------
@dataclass
class PaperRewardCfg:
    """Stage 1 reward of the contact paper: reference tracking plus regularization. Weights and the kernel widths it
    gives are the paper's; values it does not give are marked *chosen*."""

    w_base_pos: float = 3.0
    w_base_vel: float = 1.0
    w_joint_pos: float = 2.5
    w_joint_vel: float = 0.5
    w_contact_label: float = 5.0
    w_contact_pos_base: float = 2.0
    w_contact_pos_global: float = 1.0
    base_pos_std: float = 0.5
    base_ori_std: float = 1.0
    base_lin_vel_std: float = 1.0
    base_ang_vel_std: float = 3.14
    """*chosen*: BeyondMimic's body velocity widths."""
    joint_pos_std: float = 0.71
    joint_vel_std: float = 3.16
    """*chosen*: DeepMimic's pose and velocity kernels, exp(-2 sum dq^2) and exp(-0.1 sum dqd^2)."""
    contact_pos_std: float = 1.0
    """On the squared position error summed over the 9 bodies."""
    moving_speed: float = 0.1
    """Base and world contact positions count only while the reference base moves faster than this (m/s)."""
    w_action_rate: float = -0.1
    w_action_rate_2nd: float = -0.05
    w_joint_torque: float = -1e-5
    w_torque_limits: float = -1e-3
    w_dof_pos_limits: float = -10.0
    dof_pos_limits_cap: float = 1.0
    """*chosen*: cap on the summed excess (rad) beyond the soft joint limits."""
    w_dof_vel_limits: float = -1.0
    soft_dof_vel_limit: float = 0.9
    w_contact_acc: float = -0.01
    contact_acc_cap: float = 20.0
    """*chosen*: cap (m/s^2) on each contact body's acceleration norm."""
    w_planned: float = 1.5
    planned_std: float = 1.0
    """Stage 2 planned-contact term: exp(-sum_b m_b |p_b - target_b|^2 / (std^2 max(1, sum_b m_b)))."""

    @property
    def max_step_reward(self) -> float:
        """Per-step maximum of the tracking terms (15 dt)."""
        return CONTROL_DT * (self.w_base_pos + self.w_base_vel + self.w_joint_pos + self.w_joint_vel
                             + self.w_contact_label + self.w_contact_pos_base + self.w_contact_pos_global)  # fmt: skip


class PaperReward:
    """The contact paper's Stage 1 reward on world-model states, per step and multiplied by dt.

    Tracking (exp(-error^2 / std^2) kernels): base pose (world position and orientation in one kernel), base linear
    and angular velocity (one kernel), joint positions, joint velocities, the asymmetric contact-label term
    (:meth:`ContactTask.label_reward`, geometric instead of the paper's contact force), and the 9 contact bodies'
    positions in the base frame (root position and yaw, robot vs reference) and in the world. The base and world
    contact positions count only while the reference base moves.

    Regularization: action rate (first and second differences), joint torque, torque beyond the effort limit, soft
    joint position limits (capped), joint speed beyond 90% of the motor's limit (legged_gym form, at most 1 per joint),
    contact-body acceleration (capped). Torques are the G1's implicit PD law at the state after the step with the joint
    offsets of xi, clipped at the effort limit.

    Stage 2 (given the contact planner's targets and the reward schedule eta_rew): the three reference-contact
    terms weigh (1 - eta_rew) and the planned-contact term w_planned eta_rew; it pulls each body with a planned
    target (the surface point of its sphere approximation nearest the target) toward it, every step.
    """

    def __init__(self, cfg: PaperRewardCfg, ctc: ContactTask):
        self.cfg, self.ctc, self.mdp, self.motion = cfg, ctc, ctc.mdp, ctc.motion
        dev = ctc.labels.device
        self.kp, self.kd, self.effort, self.vel_limit = (torch.tensor(x, device=dev) for x in pd_params())
        self.alpha = torch.tensor([action_scale(n) for n in JOINT_NAMES], device=dev)

    def __call__(self, S, S_prev, f, v, a, a_prev, a_prev2, xi, plan_w=None, plan_valid=None, eta_rew=None) -> torch.Tensor:
        """Reward of S (reference frame f, terrain variant v) reached from S_prev by action a after a_prev, a_prev2; xi
        raw. Stage 2: the planner's world targets ``plan_w`` (B, 9, 3), their validity ``plan_valid`` (B, 9), ``eta_rew``."""
        c, m, ctc, kin = self.cfg, self.motion, self.ctc, self.mdp.kin
        p, R, vel, w, q, qd = S
        fk = kin.forward(*S)
        bp = fk[0][:, ctc.link_idx]
        rp, rR = m.root_p[f], m.root_R[f]
        ref_bp = m.body_p[f][:, ctc.link_idx]
        moving = (m.root_v[f].norm(dim=-1) > c.moving_speed).float()
        # ---- reference tracking ----
        e_base = moving * ((rp - p) ** 2).sum(-1) / c.base_pos_std**2 + rot_angle_sq(rR, R) / c.base_ori_std**2
        e_vel = ((m.root_v[f] - vel) ** 2).sum(-1) / c.base_lin_vel_std**2 + ((m.root_w[f] - w) ** 2).sum(-1) / c.base_ang_vel_std**2
        e_q = ((m.q[f] - q) ** 2).sum(-1) / c.joint_pos_std**2
        e_qd = ((m.qd[f] - qd) ** 2).sum(-1) / c.joint_vel_std**2
        e_cb = ((to_base(ref_bp, rp, rR) - to_base(bp, p, R)) ** 2).sum((-1, -2)) / c.contact_pos_std**2
        e_cg = moving * ((ref_bp - bp) ** 2).sum((-1, -2)) / c.contact_pos_std**2
        r = (c.w_base_pos * torch.exp(-e_base) + c.w_base_vel * torch.exp(-e_vel) + c.w_joint_pos * torch.exp(-e_q)
             + c.w_joint_vel * torch.exp(-e_qd))  # fmt: skip
        r_ref = (c.w_contact_label * ctc.label_reward(S, f, v, m.boxes(f, v), fk) + c.w_contact_pos_base * torch.exp(-e_cb)
                 + c.w_contact_pos_global * torch.exp(-e_cg))  # fmt: skip
        if plan_w is None:
            r = r + r_ref
        else:
            mb = plan_valid.float()
            e_pl = (mb * ((ctc.nearest_body_point(fk, plan_w) - plan_w) ** 2).sum(-1)).sum(-1)
            e_pl = e_pl / (c.planned_std**2 * mb.sum(-1).clamp(min=1))
            r = r + (1 - eta_rew) * r_ref + c.w_planned * eta_rew * torch.exp(-e_pl)
        return (r + self.regularization(S, S_prev, a, a_prev, a_prev2, xi, fk)) * CONTROL_DT

    def regularization(self, S, S_prev, a, a_prev, a_prev2, xi, fk) -> torch.Tensor:
        """The regularization terms (weighted, not multiplied by dt); ``fk`` = ``kin.forward(*S)``."""
        c, ctc, kin = self.cfg, self.ctc, self.mdp.kin
        q, qd = S[4], S[5]
        bv = fk[2][:, ctc.link_idx]
        tau_pd = self.kp * (self.mdp.default_q + xi[..., 1 : 1 + q.shape[-1]] + self.alpha * a - q) - self.kd * qd
        tau = torch.maximum(torch.minimum(tau_pd, self.effort), -self.effort)
        acc = (bv - kin.forward(*S_prev)[2][:, ctc.link_idx]) / CONTROL_DT
        q_out = (torch.relu(q - self.mdp.soft_hi) + torch.relu(self.mdp.soft_lo - q)).sum(-1)
        return (c.w_action_rate * ((a - a_prev) ** 2).sum(-1)
                + c.w_action_rate_2nd * ((a - 2 * a_prev + a_prev2) ** 2).sum(-1)
                + c.w_joint_torque * (tau**2).sum(-1)
                + c.w_torque_limits * (tau_pd - tau).abs().sum(-1)
                + c.w_dof_pos_limits * torch.clamp(q_out, max=c.dof_pos_limits_cap)
                + c.w_dof_vel_limits * (qd.abs() - c.soft_dof_vel_limit * self.vel_limit).clamp(0.0, 1.0).sum(-1)
                + c.w_contact_acc * torch.sqrt((acc**2).sum(-1) + 1e-6).clamp(max=c.contact_acc_cap).sum(-1))  # fmt: skip
