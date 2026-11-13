"""Analytic MDP heads of the contact pipeline, shared by the simulator and the world model.

Adapted from supermimic_vibed's ``supermimic/mdp.py`` (Chong Zhang, MIT), with three changes for the contact interface:

* the policy observes no reference: the last 6 observation dims are the root attitude relative to an upright frame
  at the robot's own heading (6D of R^T yaw(R)), redundant with gravity but keeping the layout;
* the world model and the critic see the height scan of the episode's terrain (:data:`terrain.CRITIC_SCAN`);
* the reference-anchored termination can accept any reference frame within ``term_window`` of the current one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from proacp.robot import NUM_J, TERM_HEIGHT
from proacp.sim.kinematics import Kinematics
from proacp.clips.library import CONTROL_DT, MotionLib
from proacp.sim.terrain import CRITIC_SCAN, grid_offsets, height_scan
from proacp.sim.rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, yaw_mat

OBS_DIM = NUM_J + NUM_J + 3 + 3 + NUM_J + 6
REF_ORI_DIMS = 6
"""The last observation dims (the attitude term), which the contact planner does not see."""
XI_DIM = 1 + NUM_J + 3  # friction, joint offsets, torso CoM offset
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J

# Table A.1 sensor noise (uniform half-widths), plus the attitude term
OBS_NOISE = torch.cat((torch.full((NUM_J,), 0.01), torch.full((NUM_J,), 0.5), torch.full((3,), 0.2), torch.full((3,), 0.05),
                       torch.zeros(NUM_J), torch.full((6,), 0.05)))  # fmt: skip
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = TERM_HEIGHT  # the robot's (robots/<name>/<name>_constants.py)
    term_tilt: float = 0.8
    term_window: int = 0
    """Frames either side of f (within its clip) whose reference root a state may match instead of frame f's: 0 =
    frame f only. Tolerates being late or early (stepping off a box after the reference) but still ends falls."""


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat((p[..., 2:3], mat_to_6d(H.transpose(-1, -2) @ R), (Rt @ v.unsqueeze(-1)).squeeze(-1),
                      (Rt @ w.unsqueeze(-1)).squeeze(-1), q, qd), -1)  # fmt: skip


INV_DIM = 1 + 6 + 3 + 3 + NUM_J + NUM_J


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1 : 1 + NUM_J] / 0.01, xi[..., 1 + NUM_J :] / 0.05), -1)


class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, default_q: torch.Tensor, cfg: RewardCfg = RewardCfg()):
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        self.obs_noise = OBS_NOISE.to(dev)
        self.scan_offsets = grid_offsets(*CRITIC_SCAN, device=dev)
        self.scan_dim = self.scan_offsets.shape[0]
        self.critic_extra = NUM_J + len(kin.tracked_idx) * 3 + 3 + 6 + XI_DIM + self.scan_dim

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi) -> torch.Tensor:
        """Noise-free proprioceptive observation (attitude relative to the robot's own heading)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1 : 1 + NUM_J]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        att = mat_to_6d(Rt @ yaw_mat(R))
        return torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, att), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    def scan(self, S, boxes: torch.Tensor) -> torch.Tensor:
        """Root height above the terrain ``boxes`` (B, K, 8) at the world model's and critic's scan points."""
        return height_scan(S[0], S[1], boxes, self.scan_offsets)

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev):
        """SuperMimic's tracking reward (already multiplied by dt) of S on reference frame f: the simulator's
        reward, which drives the start-bin sampling and the logs (the contact policy learns from its own)."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (torch.exp(-e_pos / c.body_pos_std**2) + torch.exp(-e_ori / c.body_ori_std**2) + torch.exp(-e_lv / c.body_linvel_std**2)
                   + torch.exp(-e_av / c.body_angvel_std**2) + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
                   + 0.5 * torch.exp(-e_ro / c.root_ori_std**2))  # fmt: skip
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        return (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT

    def best_frame(self, p: torch.Tensor, R: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """The reference frame within ``term_window`` of f (inside f's clip) whose root height and tilt the root
        (p, R) violates least; a state is terminated iff it violates every frame of the window."""
        m, W = self.motion, self.cfg.term_window
        c = m.clip_id[f]
        g = (f.unsqueeze(-1) + torch.arange(-W, W + 1, device=f.device)).clamp(m.clip_start[c].unsqueeze(-1), m.clip_end[c].unsqueeze(-1))
        e_z = (m.root_p[:, 2][g] - p[..., 2:3].detach()).abs() / self.cfg.term_height
        e_tilt = (m.root_R[:, 2, 2][g] - R[..., 2, 2:3].detach()).abs() / self.cfg.term_tilt
        return g.gather(-1, torch.maximum(e_z, e_tilt).argmin(-1, keepdim=True)).squeeze(-1)

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        if self.cfg.term_window:
            f = self.best_frame(p, R, f)
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts (at the best frame of the window)."""
        p, R = S[0], S[1]
        if self.cfg.term_window:
            f = self.best_frame(p, R, f)
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi_n, boxes) -> torch.Tensor:
        """Critic input: invariant state, robot-vs-reference errors, sim context and the height scan."""
        m = self.motion
        p, R, v, w, q, qd = S
        Ht = yaw_mat(R).transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi_n, self.scan(S, boxes)), -1)
