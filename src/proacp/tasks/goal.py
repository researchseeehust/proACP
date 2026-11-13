"""Stage 3 goal reaching of "Humanoid Mobility via Attention-based Contacts" (He, Wen, Zhang et al.): procedurally
generated terrains without reference motions, each with a reachable goal, and the goal-reaching task on them.

Terrains (:func:`make_terrains`): 9 box-built types from the paper's goal-reaching suite, each in 10 difficulty levels
(level 0 easiest), many random instances per level, every instance rotated by a random yaw:

==================  ==================================================================  =========================
type                course (the robot starts at the origin facing the course)            difficulty (level 0 -> 9)
==================  ==================================================================  =========================
stairs_up           4-6 steps up to a top landing; goal on the landing                  step height 5 -> 20 cm
stairs_down         start on a landing, 4-6 steps down; goal on the ground beyond       step height 5 -> 20 cm
walled_stairs       stairs up between two side walls                                     step height 5 -> 20 cm
platform            a walk-up platform of random orientation; goal on top               height 10 -> 40 cm
climb               a climbable obstacle; goal on top                                   height 40 -> 85 cm
gap                 start platform, a gap, goal platform (both 30-50 cm high)           gap 10 -> 60 cm
stepping_stones     start platform, 6 stones, goal platform (30 cm high)                stones 45 -> 25 cm wide,
                                                                                         gaps 5 -> 35 cm
balance_beam        start platform, a 1.5-2.5 m beam, goal platform (30 cm high)        beam 50 -> 20 cm wide
pallets             start platform, narrow slats with gaps, goal platform (15 cm high)  slats 20 -> 10 cm, gaps
                                                                                         5 -> 25 cm
==================  ==================================================================  =========================

Ladders and the get-up terrains of the paper are not built (no rungs to grasp, no get-up starts).

Task (:class:`GoalTask`), the paper's Table 3 goal-reaching terms with their 0/1 indicators made sigmoids (a learner
that differentiates through a world model gets no gradient from an indicator):

- move2goal (40): (1 - g_xy) g_v + g_xy g_z, with g_xy = [d_g < 0.5], g_v = [v toward goal > 0.1][|v_xy| <= v_max]
  (v_max 0.8 m/s on stairs, 2.0 elsewhere), g_z = 1 (no get-up terrains);
- standatgoal (20): g_xy g_h g_psi exp(-(d_foot + d_tilt + d_q) / 3), g_h = [|dz| < 0.2], g_psi = [|dpsi| < 0.5];
- track_planned_contact (1.5), as in Stage 2 at eta_rew = 1;
- style: flat_orientation (-10, torso), ang_vel_xy (-0.1, torso), slippage (-10, a touching foot faster than 0.5 m/s;
  geometric touch); stumbling needs contact forces and is left out;
- the Stage 1 regularizers.

Episodes last at most 15 s and end early only on a fall (the paper gives no thresholds): root less than 0.25 m above
the terrain under it, or tilt 1 - R_zz above 0.8; the G1 references go down to 0.345 m and up to 0.695 while climbing. The goal command is the paper's bounded direction d_xy / max(|d_xy|, 0.5) and the relative
heading, for the tracker and the planner alike. Success: within 0.5 m horizontally and 0.1 m vertically of the goal at
some step. Curriculum (:class:`GoalCurriculum`): per environment a fixed terrain type and a level; a successful episode
without early termination raises the level when the environment's success EMA (s <- 0.8 s + 0.2 success) exceeds 0.5;
an episode whose closest horizontal approach stays beyond 1.0 m lowers it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from proacp.robot import CONTACT_BODIES, STAND_HEIGHT, TORSO_BODY, TRACKED_BODIES
from proacp.tasks.tracking import invariant_features
from proacp.clips.library import CONTROL_DT
from proacp.tasks.contact import ContactTask, PaperReward, goal_command, to_base, wrap
from proacp.sim.terrain import BOX_FIELDS, height_scan

TERRAIN_TYPES = ("stairs_up", "stairs_down", "walled_stairs", "platform", "climb", "gap", "stepping_stones", "balance_beam", "pallets")
STAIR_TYPES = (0, 1, 2)
LEVELS = 10
MAX_BOXES = 16
FEET = [list(CONTACT_BODIES).index(b) for b in ("left_foot", "right_foot")]
TORSO = TRACKED_BODIES.index(TORSO_BODY)


# ------------------------------------------------------------------------------------------------ terrains
class _Course:
    """Boxes of one course in its canonical frame (start at the origin, facing +x)."""

    def __init__(self):
        self.boxes: list[tuple] = []

    def box(self, x0: float, x1: float, y0: float, y1: float, top: float, yaw: float = 0.0):
        """An axis-aligned box over [x0, x1] x [y0, y1] standing on the ground up to ``top``, turned by ``yaw`` about its
        center."""
        self.boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, top, yaw, (x1 - x0) / 2, (y1 - y0) / 2, top / 2))


def _course(kind: str, u: float, rng: np.random.Generator) -> tuple[_Course, float, tuple[float, float, float]]:
    """One course of ``kind`` at difficulty u in [0, 1]: its boxes, the start surface height and the goal (x, y, surface
    height) in the canonical frame."""
    c, U = _Course(), rng.uniform
    if kind in ("stairs_up", "walled_stairs"):
        h, t, n = 0.05 + 0.15 * u, U(0.28, 0.35), int(rng.integers(4, 7))
        w = U(0.8, 1.2) if kind == "walled_stairs" else U(1.0, 2.0)
        x0, L = U(1.0, 1.5), U(1.2, 1.8)
        x_end = x0 + n * t + L
        for k in range(n):
            c.box(x0 + k * t, x_end, -w / 2, w / 2, (k + 1) * h)
        if kind == "walled_stairs":
            for s in (-1, 1):
                c.box(x0 - 0.5, x_end, s * w / 2 + (0.0 if s > 0 else -0.1), s * w / 2 + (0.1 if s > 0 else 0.0), n * h + 1.0)
        return c, 0.0, (x0 + n * t + L / 2 + U(-0.2, 0.2), U(-0.15, 0.15) * w, n * h)
    if kind == "stairs_down":
        h, t, n, w = 0.05 + 0.15 * u, U(0.28, 0.35), int(rng.integers(4, 7)), U(1.0, 2.0)
        x0 = U(0.8, 1.2)
        c.box(-1.0, x0, -w / 2, w / 2, n * h)
        for k in range(1, n):
            c.box(x0 + (k - 1) * t, x0 + k * t, -w / 2, w / 2, (n - k) * h)
        return c, n * h, (x0 + (n - 1) * t + U(1.0, 1.5), U(-0.3, 0.3), 0.0)
    if kind in ("platform", "climb"):
        H = 0.1 + 0.3 * u if kind == "platform" else 0.4 + 0.45 * u
        x0, depth, w = U(1.0, 1.5), U(1.0, 2.0) if kind == "platform" else U(1.0, 1.5), U(1.0, 3.0) if kind == "platform" else U(1.5, 3.0)
        yaw = U(-0.6, 0.6) if kind == "platform" else U(-0.2, 0.2)  # the paper's platforms of randomized orientation
        c.box(x0, x0 + depth, -w / 2, w / 2, H, yaw)
        cx = x0 + depth / 2
        return c, 0.0, (cx + U(-0.15, 0.15) * depth, U(-0.15, 0.15) * w, H)
    if kind == "gap":
        H, g0, gap, w = U(0.3, 0.5), U(0.8, 1.2), 0.1 + 0.5 * u, U(1.0, 2.0)
        c.box(-1.0, g0, -w / 2, w / 2, H)
        c.box(g0 + gap, g0 + gap + 2.0, -w / 2, w / 2, H)
        return c, H, (g0 + gap + 1.0 + U(-0.3, 0.3), U(-0.2, 0.2), H)
    if kind == "stepping_stones":
        H, s, e = 0.3, 0.45 - 0.2 * u, 0.05 + 0.3 * u
        c.box(-1.0, 0.8, -0.6, 0.6, H)
        x = 0.8 + e
        for _ in range(6):
            y = U(-0.1, 0.1) * u
            c.box(x, x + s, y - s / 2, y + s / 2, H)
            x += s + e
        c.box(x, x + 1.5, -0.6, 0.6, H)
        return c, H, (x + 0.75 + U(-0.2, 0.2), U(-0.2, 0.2), H)
    if kind == "balance_beam":
        H, b, L = 0.3, 0.5 - 0.3 * u, U(1.5, 2.5)
        c.box(-1.0, 0.8, -0.6, 0.6, H)
        c.box(0.8, 0.8 + L, -b / 2, b / 2, H)
        c.box(0.8 + L, 0.8 + L + 1.5, -0.6, 0.6, H)
        return c, H, (0.8 + L + 0.75 + U(-0.2, 0.2), U(-0.2, 0.2), H)
    if kind == "pallets":
        H, th, e, w = 0.15, 0.2 - 0.1 * u, 0.05 + 0.2 * u, U(1.0, 1.5)
        c.box(-1.0, 0.8, -w / 2, w / 2, H)
        x = 0.8 + e
        while x + th < 2.8 and len(c.boxes) < MAX_BOXES - 1:
            c.box(x, x + th, -w / 2, w / 2, H)
            x += th + e
        c.box(x, x + 1.5, -w / 2, w / 2, H)
        return c, H, (x + 0.75 + U(-0.2, 0.2), U(-0.15, 0.15) * w, H)
    raise ValueError(kind)


def make_terrains(instances: int = 16, seed: int = 0) -> dict:
    """The goal-reaching terrain bank: ``instances`` random courses per type and level. Returns tensors over the G =
    types x levels x instances terrains: ``boxes`` (G, MAX_BOXES, 8), ``start`` (G, 4) (x, y, surface height, yaw),
    ``goal`` (G, 4) (x, y, pelvis height, yaw), ``type`` (G,) and ``level`` (G,); terrain index of (type, level, i) =
    (type * LEVELS + level) * instances + i."""
    rng = np.random.default_rng(seed)
    boxes = np.zeros((len(TERRAIN_TYPES) * LEVELS * instances, MAX_BOXES, BOX_FIELDS))
    start, goal, typ, lvl = [], [], [], []
    g = 0
    for ti, kind in enumerate(TERRAIN_TYPES):
        for level in range(LEVELS):
            for _ in range(instances):
                course, z0, (gx, gy, gz) = _course(kind, level / (LEVELS - 1), rng)
                assert len(course.boxes) <= MAX_BOXES, (kind, len(course.boxes))
                psi = rng.uniform(-math.pi, math.pi)  # the course's heading in the world
                cs, sn = math.cos(psi), math.sin(psi)
                for k, (cx, cy, top, yaw, hx, hy, hz) in enumerate(course.boxes):
                    boxes[g, k] = (cs * cx - sn * cy, sn * cx + cs * cy, top, math.cos(psi + yaw), math.sin(psi + yaw), hx, hy, hz)
                start.append((0.0, 0.0, z0, psi))
                goal.append((cs * gx - sn * gy, sn * gx + cs * gy, gz + STAND_HEIGHT, math.atan2(sn * gx + cs * gy, cs * gx - sn * gy)))
                typ.append(ti), lvl.append(level)
                g += 1
    t = lambda x, dt=torch.float32: torch.tensor(np.array(x), dtype=dt)  # noqa: E731
    return {"boxes": t(boxes), "start": t(start), "goal": t(goal), "type": t(typ, torch.long), "level": t(lvl, torch.long),
            "instances": instances}  # fmt: skip


class GoalCurriculum:
    """Per goal environment a fixed terrain type (round robin) and a level, moved by its episodes' outcomes."""

    def __init__(self, terrains: dict, num_envs: int, device: str, promote_ema: float = 0.5, demote_dist: float = 1.0,
                 max_start_level: int = 0):  # fmt: skip
        self.n_inst = terrains["instances"]
        self.type = torch.arange(num_envs, device=device) % len(TERRAIN_TYPES)
        self.level = torch.randint(0, max_start_level + 1, (num_envs,), device=device)
        self.success = torch.zeros(num_envs, device=device)
        self.promote_ema, self.demote_dist = promote_ema, demote_dist

    def sample(self, idx: torch.Tensor) -> torch.Tensor:
        """A random instance of each environment's (type, level): bank terrain indices."""
        i = torch.randint(0, self.n_inst, (len(idx),), device=idx.device)
        return (self.type[idx] * LEVELS + self.level[idx]) * self.n_inst + i

    def update(self, idx: torch.Tensor, success: torch.Tensor, min_dist: torch.Tensor):
        """After episodes of goal environments ``idx``: ``success`` (reached at the goal height, no early termination) and
        the closest horizontal approach ``min_dist``."""
        self.success[idx] = 0.8 * self.success[idx] + 0.2 * success.float()
        up = success & (self.success[idx] > self.promote_ema)
        down = ~success & (min_dist > self.demote_dist)
        self.level[idx] = (self.level[idx] + up.long() - down.long()).clamp(0, LEVELS - 1)


# ------------------------------------------------------------------------------------------------ task
@dataclass
class GoalCfg:
    radius: float = 0.5
    """Goal direction d_xy / max(|d_xy|, radius) of the tracker's and the planner's goal command (the paper's value)."""
    reach_xy: float = 0.5
    reach_z: float = 0.1
    """Reached (success, hindsight admission): within 0.5 m horizontally and 0.1 m vertically."""
    close_xy: float = 0.2
    """Hindsight labels end at the first step this close to the goal horizontally."""
    stand_z: float = 0.2
    stand_yaw: float = 0.5
    v_min: float = 0.1
    v_max: float = 2.0
    v_max_stairs: float = 0.8
    w_move: float = 40.0
    w_stand: float = 20.0
    w_planned: float = 1.5
    planned_std: float = 1.0
    w_flat: float = -10.0
    w_ang_vel_xy: float = -0.1
    w_slip: float = -10.0
    slip_speed: float = 0.5
    tau_xy: float = 0.05
    tau_v: float = 0.05
    tau_z: float = 0.02
    tau_yaw: float = 0.05
    """Sigmoid widths standing in for the paper's indicators (m, m/s, m, rad)."""
    fall_height: float = 0.25
    fall_tilt: float = 0.8
    tau_fall: float = 0.01
    """Fall: root height above the terrain under it (m), tilt 1 - R_zz (lying flat: 1), and the width of the soft
    version for imagination. Just beyond the references' extremes: 0.345 m and 0.695 while climbing."""
    episode_s: float = 15.0

    @property
    def max_steps(self) -> int:
        return int(round(self.episode_s / CONTROL_DT))

    @property
    def max_step_reward(self) -> float:
        return (self.w_move + self.w_stand + self.w_planned) * CONTROL_DT


class GoalTask:
    """Goal commands, reward, termination and critic input of goal reaching (shared by the simulator and imagination).
    A goal is a world (x, y, pelvis height z, heading) row; boxes are the terrain's (B, K, 8)."""

    def __init__(self, cfg: GoalCfg, ctc: ContactTask, paper_reward: PaperReward, terrains: dict):
        self.cfg, self.ctc, self.mdp, self.paper_reward = cfg, ctc, ctc.mdp, paper_reward
        dev = ctc.labels.device
        self.terrains = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in terrains.items()}
        self.origin = torch.zeros(1, 2, device=dev)
        self.critic_dim = 3 + 1 + 1 + 1  # goal command, d_g, dz, elapsed time

    def v_max(self, v: torch.Tensor) -> torch.Tensor:
        """(B,) the speed cap of terrain index v (stairs: 0.8 m/s)."""
        typ = self.terrains["type"][(v - self.ctc.V).clamp(min=0)]
        stairs = torch.zeros_like(typ, dtype=torch.bool)
        for t in STAIR_TYPES:
            stairs |= typ == t
        return torch.where(stairs, self.cfg.v_max_stairs, self.cfg.v_max)

    # ---------------- goal ----------------
    def command(self, S, goal: torch.Tensor) -> torch.Tensor:
        """(B, 3) the paper's goal command (bounded direction, relative heading) for the tracker and the planner."""
        return goal_command(S, goal[:, :3], goal[:, 3], self.cfg.radius)

    def offsets(self, S, goal: torch.Tensor):
        """Horizontal distance d_g (B,), target pelvis height minus the pelvis' dz (B,), heading error (B,)."""
        d = goal[:, :3] - S[0]
        dg = torch.sqrt((d[..., :2] ** 2).sum(-1) + 1e-6)
        return dg, d[..., 2], wrap(goal[:, 3] - torch.atan2(S[1][..., 1, 0], S[1][..., 0, 0]))

    def reached(self, S, goal: torch.Tensor) -> torch.Tensor:
        dg, dz, _ = self.offsets(S, goal)
        return (dg < self.cfg.reach_xy) & (dz.abs() < self.cfg.reach_z)

    # ---------------- termination ----------------
    def root_height(self, S, boxes: torch.Tensor) -> torch.Tensor:
        """(B,) root height above the terrain under it (differentiable in the root height)."""
        return height_scan(S[0], S[1], boxes, self.origin)[..., 0]

    def termination(self, S, boxes: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return (self.root_height(S, boxes) < c.fall_height) | (1 - S[1][..., 2, 2] > c.fall_tilt)

    def soft_termination(self, S, boxes: torch.Tensor) -> torch.Tensor:
        """Differentiable fall probability for imagined rollouts."""
        c = self.cfg
        alive = (1 - torch.sigmoid((c.fall_height - self.root_height(S, boxes)) / c.tau_fall)) * (
            1 - torch.sigmoid((1 - S[1][..., 2, 2] - c.fall_tilt) / c.tau_fall))  # fmt: skip
        return 1 - alive

    # ---------------- reward ----------------
    def _goal_terms(self, S, goal: torch.Tensor, v_max: torch.Tensor, soft: bool):
        """move2goal and the indicators of standatgoal; ``soft``: sigmoids for the indicators."""
        c = self.cfg
        v = S[2]
        dg, dz, dyaw = self.offsets(S, goal)
        d = goal[:, :2] - S[0][..., :2]
        v_to = (v[..., :2] * d).sum(-1) / dg
        speed = torch.sqrt((v[..., :2] ** 2).sum(-1) + 1e-6)
        ind = (lambda x, tau: torch.sigmoid(x / tau)) if soft else (lambda x, tau: (x > 0).float())  # noqa: E731
        g_xy = ind(c.reach_xy - dg, c.tau_xy)
        g_v = ind(v_to - c.v_min, c.tau_v) * ind(v_max - speed, c.tau_v)
        move = (1 - g_xy) * g_v + g_xy  # g_z = 1: no get-up terrains
        at = g_xy * ind(c.stand_z - dz.abs(), c.tau_z) * ind(c.stand_yaw - dyaw.abs(), c.tau_yaw)
        return move, at

    def reward(self, S, S_prev, boxes, goal, v_max, a, a_prev, a_prev2, xi, plan_w, plan_valid) -> torch.Tensor:
        """Per-step reward (multiplied by dt) of S reached from S_prev by action a; the planner's world targets ``plan_w``
        (B, 9, 3) and their validity ``plan_valid`` (B, 9)."""
        c, ctc = self.cfg, self.ctc
        R, q = S[1], S[4]
        fk = self.mdp.kin.forward(*S)
        move, at = self._goal_terms(S, goal, v_max, soft=True)
        touch = ctc.touch(S, boxes, fk)
        posture = (1 - touch[:, FEET].mean(-1)) + (1 - R[..., 2, 2] ** 2) + (q - self.mdp.default_q).abs().mean(-1)
        stand = at * torch.exp(-posture / 3)
        mb = plan_valid.float()
        e_pl = (mb * ((ctc.nearest_body_point(fk, plan_w) - plan_w) ** 2).sum(-1)).sum(-1) / (c.planned_std**2 * mb.sum(-1).clamp(min=1))
        foot_speed = fk[2][:, ctc.link_idx[FEET]].norm(dim=-1)
        slip = (touch[:, FEET] * torch.sigmoid((foot_speed - c.slip_speed) / c.tau_v)).sum(-1)
        Rt, wt = fk[1][:, TORSO], fk[3][:, TORSO]
        r = (c.w_move * move + c.w_stand * stand + c.w_planned * torch.exp(-e_pl)
             + c.w_flat * (Rt[..., 2, 0] ** 2 + Rt[..., 2, 1] ** 2) + c.w_ang_vel_xy * (wt[..., :2] ** 2).sum(-1) + c.w_slip * slip
             + self.paper_reward.regularization(S, S_prev, a, a_prev, a_prev2, xi, fk))  # fmt: skip
        return r * CONTROL_DT

    def sim_reward(self, S, goal, v_max) -> torch.Tensor:
        """For logging in the simulator: move2goal and standatgoal with the paper's indicators (dt-scaled, no posture
        factor)."""
        move, at = self._goal_terms(S, goal, v_max, soft=False)
        return (self.cfg.w_move * move + self.cfg.w_stand * at) * CONTROL_DT

    # ---------------- critic ----------------
    def critic_features(self, S, xi_n, boxes, goal, t_frac, plan_w, plan_valid) -> torch.Tensor:
        """The goal-reaching critic's input (the paper's V_fs): invariant state, sim context, height scan, the goal command,
        horizontal distance and height offset to the goal, elapsed time over the episode horizon, the planner's contacts."""
        dg, dz, _ = self.offsets(S, goal)
        tb = torch.where(plan_valid.unsqueeze(-1), to_base(plan_w, S[0], S[1]), 0.0).flatten(-2)
        return torch.cat((invariant_features(S), xi_n, self.mdp.scan(S, boxes), self.command(S, goal), dg.unsqueeze(-1),
                          dz.unsqueeze(-1), t_frac.unsqueeze(-1), tb), -1)  # fmt: skip
