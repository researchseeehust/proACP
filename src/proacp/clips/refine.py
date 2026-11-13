"""Clean the retargeted clips the way the paper's retargeting does, without retargeting them again.

The clips leave planted feet centimeters above or inside the box tops and palms up to 9 cm inside them, so no contact
threshold on their geometry is right. Each clip is corrected by a minimal whole-trajectory IK over root pose and joints:
the tracked bodies stay on the source, minimum-distance constraints keep the contact bodies out of the terrain, and
temporal regularization keeps the correction smooth. In addition, a body in a contact of the source touches the terrain
exactly (distance 0), so that the labels of the result need no tolerance.

Contacts of the source (:func:`contact_segments`): a body is in contact within ``CONTACT_DIST`` or while it hardly moves
within ``SUPPORT_DIST`` of a surface. Gaps are filled, a foot that stays in place across a gap (rolling from the heel to the
toe) stays in one contact, and contacts shorter than ``MIN_FRAMES`` are dropped.

Output: the source layout and fields, ``<out>/<folder>/<clip>_motion.npz`` + ``_terrain.npy``; the optimization runs in
the clip's first terrain variant.

    python -m proacp.clips.labels                                    # the source's contacts
    python -m proacp.clips.refine --out data/php_parkour_refined
    python -m proacp.clips.labels --motion_dir data/php_parkour_refined
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time

import mujoco
import numpy as np
import torch

from proacp.sim.rotations import mat_to_quat, quat_to_mat, so3_exp
from proacp.robot import CONTACT_BODIES, MOTION_DIR, MOTION_GLOB, NB, TRACKED_BODIES, get_spec
from proacp.sim.kinematics import Kinematics
from proacp.clips.library import load_php_npz
from proacp.tasks.contact import BodySpheres
from proacp.sim.terrain import box_sdf, load_terrain

CONTACT_DIST = (0.008, 0.008, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005)
"""Per contact body (m): touching in any motion; a toe leaving the ground drags a few millimeters above it."""
SUPPORT_DIST = (0.03, 0.03, 0.012, 0.012, 0.03, 0.03, 0.012, 0.012, 0.005)
"""Hardly moving this close to a surface is a contact: ground and box tops, and box sides, which a climbing foot pushes
against. Knees and elbows only on boxes and only just above them: the shin of a robot standing on a box is 2-4 cm off it,
and pulling it down swings the whole leg."""
SUPPORT_SPEED = 0.5
MIN_FRAMES = (4, 4, 10, 10, 10, 10, 10, 10, 10)
"""Shorter contacts are dropped; a real footfall can last 0.1 s (the paper's 0.2 s is for the planner's hindsight labels)."""
FILL_GAP = 3
ROLL_GAP, ROLL_XY, ROLL_RISE = 15, 0.08, 0.04  # one contact across a gap (a foot rolling heel to toe): frames, m, m
BOX_ONLY = (2, 3, 6, 7)
FEET = (0, 1)


def _runs(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.diff(np.r_[0, c.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def contact_segments(dist: np.ndarray, surface: np.ndarray, pos: np.ndarray, fps: float) -> np.ndarray:
    """(T, 9) bool contacts of a source clip from its distances (T, 9), nearest surfaces (T, 9; 0 ground, k >= 1 box) and
    link positions (T, 9, 3)."""
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=-1) * fps
    out = np.zeros(dist.shape, bool)
    for b in range(NB):
        near = surface[:, b] > 0 if b in BOX_ONLY else surface[:, b] >= 0
        c = (dist[:, b] <= CONTACT_DIST[b]) | ((dist[:, b] <= SUPPORT_DIST[b]) & (speed[:, b] < SUPPORT_SPEED) & near)
        c &= dist[:, b] != 0.0  # exactly 0: mj_geomDistance degenerate at a box edge, not a touch
        s, e = _runs(c)
        for e0, s1 in zip(e[:-1], s[1:]):
            p0, p1 = pos[e0 - 1, b], pos[s1, b]
            same = (b in FEET and s1 - e0 <= ROLL_GAP and np.linalg.norm(p1[:2] - p0[:2]) <= ROLL_XY
                    and pos[e0:s1, b, 2].max() - max(p0[2], p1[2]) <= ROLL_RISE)  # fmt: skip
            if s1 - e0 <= FILL_GAP or same:
                c[e0:s1] = True
        for s0, e0 in zip(*_runs(c)):
            if e0 - s0 < MIN_FRAMES[b]:
                c[s0:e0] = False
        out[:, b] = c
    return out


class Refiner:
    def __init__(self, device: str, iters: int, lr: float, w_contact: float, w_clear: float, w_smooth: float):
        self.dev, self.iters, self.lr = device, iters, lr
        self.w_contact, self.w_clear, self.w_smooth = w_contact, w_clear, w_smooth
        self.kin = Kinematics(device)
        sph = BodySpheres()
        self.pts, self.rad = sph.pts.to(device), sph.rad.to(device)
        frame = [TRACKED_BODIES.index(v[0]) for v in CONTACT_BODIES.values()]
        self.frame = torch.tensor(frame, device=device)[sph.owner.to(device)]
        self.bounds = sph.bounds

    def fk(self, p, R, q):
        zero = torch.zeros_like(p)
        return self.kin.forward(p, R, zero, zero, q, torch.zeros_like(q))[:2]

    def distances(self, P, Rs, boxes):
        """(T, S) signed distance of every sphere to the ground and ``boxes`` (K, 8)."""
        x = P[:, self.frame] + (Rs[:, self.frame] @ self.pts.unsqueeze(-1)).squeeze(-1)
        d = x[..., 2]
        for k in range(len(boxes)):
            d = torch.minimum(d, box_sdf(x, boxes[k : k + 1].expand(len(x), -1)))
        return d - self.rad

    def refine(self, pos, quat, q, boxes, contact):
        dev, iters = self.dev, self.iters if len(boxes) else min(self.iters, 600)
        p0, R0, q0 = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (pos, quat, q))
        R0 = quat_to_mat(R0)
        boxes = torch.tensor(boxes, dtype=torch.float32, device=dev)
        c = torch.tensor(contact, device=dev)
        with torch.no_grad():
            P0 = self.fk(p0, R0, q0)[0]
        dp, dr, dq = (torch.zeros(*s, device=dev, requires_grad=True) for s in ((len(p0), 3), (len(p0), 3), q0.shape))
        opt = torch.optim.Adam((dp, dr, dq), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        q_lo, q_hi = self.kin.q_lo, self.kin.q_hi
        for _ in range(iters):
            R = R0 @ so3_exp(dr)
            qn = torch.clamp(q0 + dq, q_lo, q_hi)
            P, Rs = self.fk(p0 + dp, R, qn)
            d = self.distances(P, Rs, boxes)
            body = torch.stack([d[:, s:e].min(-1).values for s, e in self.bounds], -1)
            track = ((P - P0) ** 2).sum(-1).mean() + 0.01 * (dq**2).mean() + 0.01 * (dr**2).mean()
            clear = (torch.relu(-d) ** 2).sum(-1).mean()
            touch = (body[c] ** 2).sum() / max(len(p0), 1)
            smooth = sum(((x[1:] - x[:-1]) ** 2).mean() + 10 * ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).mean() for x in (dp, dr, dq))
            loss = track + self.w_clear * clear + self.w_contact * touch + self.w_smooth * smooth
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            R = R0 @ so3_exp(dr)
            qn = torch.clamp(q0 + dq, q_lo, q_hi)
            pn = p0 + dp
            P, Rs = self.fk(pn, R, qn)
            d = self.distances(P, Rs, boxes)
            body = torch.stack([d[:, s:e].min(-1).values for s, e in self.bounds], -1)
            e = (P - P0).norm(dim=-1)
            stats = {"body_err_mean": e.mean().item(), "body_err_max": e.max().item(), "penetration_max": (-d.min()).clamp(min=0).item(),
                     "touch_err_p95": body[c].abs().quantile(0.95).item() if c.any() else 0.0, "contact_frames": int(c.sum()),
                     "touch_err_p95_per_body": [round(body[:, b][c[:, b]].abs().quantile(0.95).item(), 4) if c[:, b].any() else 0.0 for b in range(NB)]}  # fmt: skip
        return pn.cpu().numpy(), mat_to_quat(R).cpu().numpy(), qn.cpu().numpy(), stats


def quat_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q.astype(np.float64))
    return R.reshape(3, 3)


def ang_vel_world(quat: np.ndarray, fps: float) -> np.ndarray:
    """(T, 4) wxyz -> (T, 3) world angular velocity, central differences (one-sided at the ends)."""
    n = len(quat)
    out = np.zeros((n, 3))
    for t in range(n):
        i0, i1 = max(t - 1, 0), min(t + 1, n - 1)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, quat[i1], quat[i0] * np.array((1.0, -1.0, -1.0, -1.0)))
        v = np.zeros(3)
        mujoco.mju_quat2Vel(v, q, 1.0)
        out[t] = v * fps / max(i1 - i0, 1)
    return out


def write_clip(model: mujoco.MjModel, src_path: str, out_path: str, pos, quat, q):
    """The source's fields for the new trajectory; velocities are central differences, as in the source."""
    src = dict(np.load(src_path))
    fps = float(np.asarray(src["fps"]).reshape(-1)[0])
    jp = np.concatenate((pos, quat, q), -1).astype(np.float64)
    names = [str(n) for n in src["body_names"]]
    ids = {n: model.body(n).id for n in names if not n.endswith("_foot_contact_point")}
    off = {}
    for side in ("left", "right"):  # the contact point keeps its fixed offset in the ankle roll link
        a, c = names.index(f"{side}_ankle_roll_link"), names.index(f"{side}_foot_contact_point")
        off[side] = quat_mat(src["body_quat_w"][0, a]).T @ (src["body_pos_w"][0, c] - src["body_pos_w"][0, a])
    data = mujoco.MjData(model)
    T = len(jp)
    bp, bq = np.zeros((T, len(names), 3)), np.zeros((T, len(names), 4))
    for t in range(T):
        data.qpos[:] = jp[t]
        mujoco.mj_kinematics(model, data)
        for k, n in enumerate(names):
            if n in ids:
                bp[t, k], bq[t, k] = data.xpos[ids[n]], data.xquat[ids[n]]
            else:
                side = n.split("_")[0]
                a = ids[f"{side}_ankle_roll_link"]
                bp[t, k], bq[t, k] = data.xpos[a] + data.xmat[a].reshape(3, 3) @ off[side], data.xquat[a]
    ang = np.stack([ang_vel_world(bq[:, k], fps) for k in range(len(names))], 1)
    jv = np.concatenate((np.gradient(jp[:, :3], axis=0) * fps, ang[:, 0], np.gradient(jp[:, 7:], axis=0) * fps), -1)
    out = {**src, "joint_pos": jp, "joint_vel": jv, "body_pos_w": bp, "body_quat_w": bq, "body_lin_vel_w": np.gradient(bp, axis=0) * fps, "body_ang_vel_w": ang}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    shutil.copy(src_path.replace("_motion.npz", "_terrain.npy"), out_path.replace("_motion.npz", "_terrain.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion_dir", default=MOTION_DIR)
    ap.add_argument("--contacts_dir", default=None, help="clips.labels output of the source (default: <motion_dir>_contacts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500, help="600 on clips without boxes: nothing there conflicts")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w_contact", type=float, default=300.0)
    ap.add_argument("--w_clear", type=float, default=1000.0)
    ap.add_argument("--w_smooth", type=float, default=30.0)
    ap.add_argument("--clips", type=str, default=None, help="only clips whose name contains this (testing)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cdir = args.contacts_dir or args.motion_dir.rstrip("/") + "_contacts"
    files = sorted(glob.glob(os.path.join(args.motion_dir, MOTION_GLOB)))
    if args.clips:
        files = [f for f in files if args.clips in f]
    rf = Refiner(args.device, args.iters, args.lr, args.w_contact, args.w_clear, args.w_smooth)
    model = get_spec().compile()
    report, t0 = {}, time.time()
    for i, f in enumerate(files):
        rel = os.path.relpath(f, args.motion_dir)[: -len("_motion.npz")]
        lab = np.load(os.path.join(cdir, rel + ".npz"))
        pos, quat, q, fps = load_php_npz(f)
        boxes = load_terrain(f)[0]
        boxes = boxes[boxes[:, 5] > 0]
        c = contact_segments(lab["contact_dist"][0], lab["contact_surface"][0], lab["body_pos_w"], fps)
        pn, qt, qn, st = rf.refine(pos, quat, q, boxes, c)
        write_clip(model, f, os.path.join(args.out, rel + "_motion.npz"), pn, qt, qn)
        report[rel] = st
        print(f"{i + 1}/{len(files)} {rel}: " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in st.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(args.out, f"refine_{args.clips or 'all'}.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
