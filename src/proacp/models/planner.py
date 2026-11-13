"""Contact planner of "Humanoid Mobility via Attention-based Contacts" (He, Wen, Zhang et al.) on a height scan.

For each of the 9 :data:`proacp.robot.CONTACT_BODIES` the planner predicts where its next new contact within
1.5 s will be: affordance weights over the 17 x 11 points of a local height scan (the top-1 point is the target) and
a contact occupancy (the target is kept above 0.5, zeros otherwise). Inputs: five steps of proprioception, the bounded
goal direction and heading, and the scan points, all in the base frame (root position, yaw). The scan is exact: the
terrain height under each grid point from the episode's boxes (:func:`proacp.sim.terrain.height_scan`), with
Gaussian noise on what the planner sees. The network is the paper's Table S3 attention planner.

It learns from hindsight labels: successful episodes of the tracker (started within the first 10% of their clip,
reached 90% of it) are scanned backward for each body's next contact onset (contacts of at least 10 steps), and the
terrain point touched there, in the base frame of each step, is that step's label if it lies inside the scan
footprint. Contacts are geometric (the sphere approximation of :class:`proacp.tasks.contact.ContactTask` within
``touch_dist``), as the tracker's labels; the paper measures contact forces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proacp.robot import NB
from proacp.tasks.tracking import REF_ORI_DIMS, unflatten_state
from proacp.clips.library import CONTROL_DT
from proacp.models.networks import RunningNorm
from proacp.tasks.contact import ContactTask, debounce, drop_short, from_base, goal_command, next_onsets, to_base
from proacp.sim.terrain import grid_offsets, height_scan


@dataclass
class PlannerCfg:
    scan_x: tuple[float, float, int] = (-0.8, 0.8, 17)
    scan_y: tuple[float, float, int] = (-0.5, 0.5, 11)
    """Scan grid (start, end, count) in the base frame; also the footprint targets must lie in."""
    hist: int = 5
    goal_radius: float = 0.3
    """Goal direction d_xy / max(|d_xy|, goal_radius) (the paper's reference-following value)."""
    scan_noise: float = 0.03
    """Gaussian noise (m) on the scan heights the planner sees."""
    sigma_a: float = 0.1
    """Bandwidth (m) of the soft affordance target (Eq. 1)."""
    occ_threshold: float = 0.5
    horizon_s: float = 1.5
    min_contact_steps: int = 10
    start_max: float = 0.1
    reach_min: float = 0.9
    """Success filter: start within this fraction of the clip, reach at least this fraction."""
    lr: float = 1e-4
    batch: int = 4096
    sub_batches: int = 8
    min_entries: int = 1000
    capacity: int = 1_000_000


class PlannerScan:
    """The planner's terrain input: grid points of the base frame, terrain height relative to the root."""

    def __init__(self, cfg: PlannerCfg, device: str):
        self.offsets = grid_offsets(cfg.scan_x, cfg.scan_y, device=device)  # (P, 2), x-major
        self.L, self.W = cfg.scan_x[2], cfg.scan_y[2]
        self.half = (max(abs(cfg.scan_x[0]), abs(cfg.scan_x[1])), max(abs(cfg.scan_y[0]), abs(cfg.scan_y[1])))
        self.num_z = self.offsets.shape[0]

    def heights(self, S, boxes: torch.Tensor) -> torch.Tensor:
        """(B, P) terrain height minus root height at the grid points of states S on ``boxes`` (B, K, 8)."""
        return -height_scan(S[0].detach(), S[1].detach(), boxes, self.offsets)

    @staticmethod
    def noisy(z: torch.Tensor, sigma: float) -> torch.Tensor:
        return z + sigma * torch.randn_like(z)

    def points(self, z: torch.Tensor) -> torch.Tensor:
        """(B, P, 3) base-frame scan points of heights z (B, P)."""
        return torch.cat((self.offsets.expand(z.shape[0], -1, -1), z.unsqueeze(-1)), -1)

    def inside(self, x: torch.Tensor) -> torch.Tensor:
        """Base-frame points x (..., 3) within the footprint."""
        return (x[..., 0].abs() <= self.half[0] + 1e-6) & (x[..., 1].abs() <= self.half[1] + 1e-6)


class ContactPlanner(nn.Module):
    """Table S3 attention planner. Terrain points: a per-point linear embedding and a two-layer convolution over
    the height grid, projected to keys/values; a pointwise MLP max-pooled into a terrain context. Proprioception,
    goal and context: a shared vector, modulated per body by FiLM from learned body embeddings into 9 queries.
    One cross-attention layer: its head-averaged weights are the affordances; its output with the context gives
    the occupancy."""

    def __init__(self, prop_dim: int, cfg: PlannerCfg, d: int = 96, heads: int = 32):
        super().__init__()
        self.prop_dim, self.hist = prop_dim, cfg.hist
        self.L, self.W = cfg.scan_x[2], cfg.scan_y[2]
        self.point = nn.Sequential(nn.Linear(3, 16), nn.ELU())
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ELU(), nn.Conv2d(16, 48, 5, padding=2), nn.ELU())
        self.proj = nn.Linear(64, d)
        self.ctx = nn.Sequential(nn.Linear(d, 64), nn.ELU(), nn.Linear(64, 64))
        self.shared = nn.Sequential(nn.Linear(prop_dim * cfg.hist + 3 + 64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(128, d))  # fmt: skip
        self.body = nn.Parameter(0.1 * torch.randn(NB, d))
        self.film = nn.Sequential(nn.Linear(d, 64), nn.ELU(), nn.Linear(64, 2 * d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.occ = nn.Sequential(nn.Linear(d + 64, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(),
                                 nn.Linear(512, 1))  # fmt: skip
        self.prop_norm = RunningNorm(prop_dim)

    def forward(self, prop: torch.Tensor, goal: torch.Tensor, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """prop (B, hist, prop_dim), goal (B, 3), pts (B, P, 3) -> log affordances (B, 9, P), occupancy logits (B, 9)."""
        B = pts.shape[0]
        g = self.conv(pts[..., 2].reshape(B, 1, self.L, self.W)).flatten(2).transpose(1, 2)  # (B, P, 48)
        kv = self.proj(torch.cat((self.point(pts), g), -1))
        ctx = self.ctx(kv).max(1).values
        h = self.shared(torch.cat((self.prop_norm(prop).flatten(1), goal, ctx), -1))
        scale, shift = self.film(self.body).chunk(2, -1)
        out, w = self.attn(h.unsqueeze(1) * (1 + scale) + shift, kv, kv, need_weights=True, average_attn_weights=True)
        occ = self.occ(torch.cat((out, ctx.unsqueeze(1).expand(-1, NB, -1)), -1)).squeeze(-1)
        return torch.log(w.clamp_min(1e-12)), occ


def proprio(o: torch.Tensor) -> torch.Tensor:
    """The planner's proprioception from tracker observations (..., obs_dim): all but the attitude term."""
    return o[..., :-REF_ORI_DIMS]


def planner_goal(ctc: ContactTask, S, f: torch.Tensor, goal_steps: torch.Tensor, radius: float) -> torch.Tensor:
    """(B, 3) bounded goal direction and relative heading of the reference ``goal_steps`` ahead (the tracker's goal)."""
    return ctc.reference_goal(S, f, goal_steps, radius)


def decode(log_alpha: torch.Tensor, occ_logit: torch.Tensor, pts: torch.Tensor, threshold: float):
    """Top-1 points (B, 9, 3) of the scan ``pts``, zeros where the occupancy is at most ``threshold``; and validity."""
    i = log_alpha.argmax(-1)
    tgt = pts.gather(1, i.unsqueeze(-1).expand(-1, -1, 3))
    valid = torch.sigmoid(occ_logit) > threshold
    return torch.where(valid.unsqueeze(-1), tgt, 0.0), valid


@torch.no_grad()
def plan(planner: ContactPlanner, scan: PlannerScan, cfg: PlannerCfg, ctc: ContactTask, o_hist: torch.Tensor, S,
         f: torch.Tensor, v: torch.Tensor, goal_steps: torch.Tensor | None, noise: bool = True, goal: torch.Tensor | None = None):  # fmt: skip
    """The planner on state S (reference frame f, terrain v): world targets (B, 9, 3) (zeros where invalid) and
    validity (B, 9). ``o_hist`` (B, >= hist, obs_dim) are the tracker's observations, newest last. ``goal`` (B, 3): the
    planner's goal command (goal reaching) instead of the reference's."""
    p, R = S[0].detach(), S[1].detach()
    z = scan.heights((p, R), ctc.motion.boxes(f, v))
    if noise:
        z = scan.noisy(z, cfg.scan_noise)
    pts = scan.points(z)
    with torch.autocast("cuda", dtype=torch.float16):  # inference only, ~4x faster compiled; bf16 flips top-1 points
        g = planner_goal(ctc, (p, R), f, goal_steps, cfg.goal_radius) if goal is None else goal
        la, ol = planner(proprio(o_hist[:, -cfg.hist :]), g, pts)
    tgt, valid = decode(la.float(), ol.float(), pts, cfg.occ_threshold)
    return torch.where(valid.unsqueeze(-1), from_base(tgt, p, R), 0.0), valid


def planner_loss(log_alpha: torch.Tensor, occ_logit: torch.Tensor, pts: torch.Tensor, tgt: torch.Tensor, occ: torch.Tensor,
                 sigma_a: float) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Eq. 2: cross-entropy to the soft target of Eq. 1 (over the valid contacts) and occupancy BCE (all bodies)."""
    d2 = ((pts.unsqueeze(1) - tgt.unsqueeze(2)) ** 2).sum(-1)  # (B, 9, P)
    soft = torch.softmax(-d2 / (2 * sigma_a**2), -1)
    ce = -(soft * log_alpha).sum(-1)
    occ = occ.float()
    return (ce * occ).sum() / occ.sum().clamp(min=1), F.binary_cross_entropy_with_logits(occ_logit, occ)


# ---------------------------------------------------------------- hindsight labels
def onset_index(touch: np.ndarray, min_steps: int, horizon: int) -> np.ndarray:
    """(T, 9) step of each body's next contact onset after t within ``horizon`` steps (-1: none), from per-step touch
    (T, 9): gaps of up to 2 steps filled, then contacts shorter than ``min_steps`` dropped."""
    return next_onsets(drop_short(debounce(touch, 2), min_steps), horizon)


def hindsight_labels(on: torch.Tensor, cp: torch.Tensor, p: torch.Tensor, R: torch.Tensor, scan: PlannerScan):
    """Labels of each step: the touched point ``cp`` (N, T, 9, 3) world at the onset step ``on`` (N, T, 9) in the base
    frame of the step's root (p (N, T, 3), R (N, T, 3, 3)); occupancy where there is an onset inside the footprint."""
    N, T = on.shape[:2]
    idx = on.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, 3)
    tgt = to_base(cp.gather(1, idx).flatten(0, 1), p.flatten(0, 1), R.flatten(0, 1)).view(N, T, NB, 3)
    occ = (on >= 0) & scan.inside(tgt)
    return torch.where(occ.unsqueeze(-1), tgt, 0.0), occ


@torch.no_grad()
def label_episodes(buf, envs: torch.Tensor, rows_end: torch.Tensor, lens: torch.Tensor, ctc: ContactTask, scan: PlannerScan,
                   cfg: PlannerCfg, success: torch.Tensor | None = None, keep: torch.Tensor | None = None,
                   goal_radius: float = 0.5) -> tuple[dict | None, int]:  # fmt: skip
    """Planner samples of the finished episodes of ``envs`` still in the replay buffer ``buf`` (layout (time, env)):
    each ended at storage row ``rows_end`` after ``lens`` steps. Admits the successful ones; returns their per-step
    samples (or None) and how many were admitted.

    Goal reaching (the paper's Stage 3): ``success`` (per episode: the goal was reached) replaces the reference-progress
    filter, only the first ``keep`` steps of each episode are labeled, and the planner's goal input comes from the
    buffer's ``goal`` field (world x, y, z, yaw) with radius ``goal_radius``."""
    m, dev = ctc.motion, lens.device
    lens = lens.long()
    k = torch.arange(int(lens.max()), device=dev)
    kk = torch.minimum(k, lens.unsqueeze(-1) - 1)  # steps past an episode's end repeat its last one
    rows = (rows_end.unsqueeze(-1) - lens.unsqueeze(-1) + 1 + kk) % buf.T
    e2 = envs.unsqueeze(-1).expand_as(rows)
    f = buf.data["f"][rows, e2]
    if success is None:
        c = m.clip_id[f[:, 0]]
        span = (m.clip_len[c] - 1).float()
        start = (f[:, 0] - m.clip_start[c]).float() / span
        reach = (f[:, -1] + 1 - m.clip_start[c]).float() / span
        success = (start <= cfg.start_max) & (reach >= cfg.reach_min)
    ok = success.nonzero(as_tuple=False).squeeze(-1)
    if len(ok) == 0:
        return None, 0
    rows, e2, f, lens = rows[ok], e2[ok], f[ok], lens[ok]
    if keep is not None:  # the labeled segment ends early: steps past it are dropped below
        lens = torch.minimum(lens, keep.long()[ok])
    E, T = f.shape
    S = unflatten_state(buf.data["s"][rows, e2].flatten(0, 1))
    ff, vv = f.flatten(), buf.data["v"][rows, e2].flatten()
    boxes = m.boxes(ff, vv)
    fk = ctc.mdp.kin.forward(*S)
    touch = (ctc.body_distance(S, boxes, fk) <= ctc.cfg.touch_dist).view(E, T, NB).cpu().numpy()
    cp = ctc.contact_point(S, boxes, fk).view(E, T, NB, 3)
    H = int(round(cfg.horizon_s / CONTROL_DT))
    on = np.full((E, T, NB), -1, np.int64)
    for e, n in enumerate(lens.tolist()):
        on[e, :n] = onset_index(touch[e, :n], cfg.min_contact_steps, H)
    p, R = S[0].view(E, T, 3), S[1].view(E, T, 3, 3)
    tgt, occ = hindsight_labels(torch.tensor(on, device=dev), cp, p, R, scan)
    # planner inputs of every step: proprioception history (repeating the first step), goal, clean scan heights
    o = buf.data["o"][rows, e2]
    h = (k[:T].unsqueeze(-1) + torch.arange(-cfg.hist + 1, 1, device=dev)).clamp(min=0)  # (T, hist)
    prop = proprio(o[:, h])  # (E, T, hist, prop_dim)
    if keep is None:
        gs = buf.data["goal_steps"][rows[:, -1], e2[:, -1]].repeat_interleave(T)
        goal = planner_goal(ctc, S, ff, gs, cfg.goal_radius)
    else:
        gw = buf.data["goal"][rows, e2].flatten(0, 1)
        goal = goal_command(S, gw[:, :3], gw[:, 3], goal_radius)
    z = scan.heights(S, boxes)
    valid = (k[:T].unsqueeze(0) < lens.unsqueeze(-1)).flatten()
    out = {"prop": prop.flatten(0, 1), "goal": goal, "z": z, "tgt": tgt.flatten(0, 1), "occ": occ.flatten(0, 1), "clip": m.clip_id[ff],
           "v": vv}  # fmt: skip
    return {key: x[valid] for key, x in out.items()}, len(ok)


class PlannerBuffer:
    """FIFO of hindsight samples on the GPU (proprioception and scan heights in fp16, scan noise added when sampled)."""

    def __init__(self, capacity: int, hist: int, prop_dim: int, num_points: int, device: str):
        self.cap, self.device = capacity, device
        self.data = {
            "prop": torch.zeros(capacity, hist, prop_dim, dtype=torch.float16, device=device),
            "goal": torch.zeros(capacity, 3, device=device),
            "z": torch.zeros(capacity, num_points, dtype=torch.float16, device=device),
            "tgt": torch.zeros(capacity, NB, 3, device=device),
            "occ": torch.zeros(capacity, NB, dtype=torch.bool, device=device),
            "clip": torch.zeros(capacity, dtype=torch.long, device=device),
            "v": torch.zeros(capacity, dtype=torch.long, device=device),
        }
        self.ptr, self.size, self.added = 0, 0, 0

    def add(self, d: dict):
        n = len(d["occ"])
        if n == 0:
            return
        if n > self.cap:
            d, n = {k: v[-self.cap :] for k, v in d.items()}, self.cap
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.cap
        for k, v in self.data.items():
            v[idx] = d[k].to(v.dtype)
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)
        self.added += n

    def sample(self, B: int) -> dict:
        idx = torch.randint(0, self.size, (B,), device=self.device)
        return {k: v[idx] for k, v in self.data.items()}


class GroupedPlannerBuffer:
    """Planner buffers per group (the paper's goal-reaching buffer, partitioned by terrain type), sampled uniformly over
    the groups that hold samples, then uniformly within the group."""

    def __init__(self, groups: int, capacity: int, hist: int, prop_dim: int, num_points: int, device: str):
        self.bufs = [PlannerBuffer(capacity // groups, hist, prop_dim, num_points, device) for _ in range(groups)]
        self.device = device

    @property
    def size(self) -> int:
        return sum(b.size for b in self.bufs)

    def add(self, d: dict, group: torch.Tensor):
        for g in group.unique().tolist():
            sel = group == g
            self.bufs[g].add({k: v[sel] for k, v in d.items()})

    def sample(self, B: int) -> dict:
        full = [b for b in self.bufs if b.size > 0]
        which = torch.randint(0, len(full), (B,))
        parts = [b.sample(n) for b, n in zip(full, torch.bincount(which, minlength=len(full)).tolist()) if n > 0]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


class MixedPlannerBuffer:
    """Planner buffers sampled as one, each in proportion to the samples it holds (the paper's Stage 3: the
    reference-following and goal-reaching buffers)."""

    def __init__(self, *bufs):
        self.bufs, self.device = bufs, bufs[0].device

    @property
    def size(self) -> int:
        return sum(b.size for b in self.bufs)

    def sample(self, B: int) -> dict:
        which = torch.multinomial(torch.tensor([float(b.size) for b in self.bufs]), B, replacement=True)
        parts = [b.sample(n) for b, n in zip(self.bufs, torch.bincount(which, minlength=len(self.bufs)).tolist()) if n > 0]
        perm = torch.randperm(B, device=self.device)  # mixed sub-batches in planner_update
        return {k: torch.cat([p[k] for p in parts])[perm] for k in parts[0]}


def planner_batch_loss(planner: ContactPlanner, b: dict, scan: PlannerScan, cfg: PlannerCfg, noise: bool = True):
    """Losses of buffer samples ``b``: inputs with fresh scan noise, soft targets on the clean scan."""
    z = b["z"].float()
    la, ol = planner(b["prop"].float(), b["goal"], scan.points(scan.noisy(z, cfg.scan_noise) if noise else z))
    return planner_loss(la, ol, scan.points(z), b["tgt"], b["occ"], cfg.sigma_a), (la, ol)


def planner_update(planner: ContactPlanner, opt: torch.optim.Optimizer, pbuf: PlannerBuffer, scan: PlannerScan,
                   cfg: PlannerCfg) -> torch.Tensor:  # fmt: skip
    """One optimizer update on ``cfg.batch`` samples in ``cfg.sub_batches`` accumulated parts; (affordance, occupancy)."""
    batch = pbuf.sample(cfg.batch)
    sb = cfg.batch // cfg.sub_batches
    opt.zero_grad(set_to_none=True)
    tot = torch.zeros(2, device=pbuf.device)
    for i in range(cfg.sub_batches):
        (l_aff, l_occ), _ = planner_batch_loss(planner, {k: v[i * sb : (i + 1) * sb] for k, v in batch.items()}, scan, cfg)
        ((l_aff + l_occ) / cfg.sub_batches).backward()
        tot += torch.stack((l_aff.detach(), l_occ.detach())) / cfg.sub_batches
    opt.step()
    return tot


@torch.no_grad()
def planner_metrics(planner: ContactPlanner, b: dict, scan: PlannerScan, cfg: PlannerCfg, chunk: int = 4096) -> dict:
    """Losses and, per body, occupancy precision / recall / F1, median top-1 error (m) and the share within 0.1 m
    (over the samples with a label) of buffer samples ``b``, ``chunk`` samples at a time."""
    z = b["z"].float()
    pts_in = scan.points(scan.noisy(z, cfg.scan_noise))  # as planner_batch_loss
    outs = [planner(b["prop"][i : i + chunk].float(), b["goal"][i : i + chunk], pts_in[i : i + chunk]) for i in range(0, len(z), chunk)]
    la, ol = torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
    pts = scan.points(z)  # top-1 error against the exact geometry
    l_aff, l_occ = planner_loss(la, ol, pts, b["tgt"], b["occ"], cfg.sigma_a)
    pred = torch.sigmoid(ol) > cfg.occ_threshold
    occ = b["occ"]
    tp, n_pred, n_lab = (pred & occ).sum(0).float(), pred.sum(0).float(), occ.sum(0).float()
    prec, rec = tp / n_pred.clamp(min=1), tp / n_lab.clamp(min=1)
    top1 = pts.gather(1, la.argmax(-1).unsqueeze(-1).expand(-1, -1, 3))
    err = (top1 - b["tgt"]).norm(dim=-1)
    med = [err[occ[:, i], i].median().item() if occ[:, i].any() else float("nan") for i in range(NB)]
    near = [(err[occ[:, i], i] <= 0.1).float().mean().item() if occ[:, i].any() else float("nan") for i in range(NB)]
    return {"aff": l_aff.item(), "occ": l_occ.item(), "precision": prec.tolist(), "recall": rec.tolist(),
            "f1": (2 * prec * rec / (prec + rec).clamp(min=1e-9)).tolist(), "median_err": med, "within_0.1": near,
            "n_label": n_lab.tolist()}  # fmt: skip
