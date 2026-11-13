"""Reference motion library of the PHP parkour clips (G1, 50 fps), with FK of the tracked bodies and the boxes.

Motion files (``<clip>_motion.npz``): ``joint_pos`` (T, 36) = root position, root quaternion (wxyz), the 29
joint angles in ``joint_names`` order, plus ``fps``. Clip names are relative to the motion directory
(``motions-low-step/forward_walk0.5s_step0.5s``). Each clip's boxes in all their variants are
``clip_boxes`` (C, V, K, 8); ``variant_ok`` (C, V) marks the variants the reference motion does not pass
through (no body origin more than :data:`VARIANT_TOL` below the terrain under it), the only ones episodes use.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from proacp.robot import JOINT_NAMES, MOTION_GLOB
from proacp.sim.kinematics import Kinematics
from proacp.sim.terrain import box_heights, load_clip_boxes
from proacp.sim.rotations import quat_slerp, quat_to_mat, so3_log

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
VARIANT_TOL = 0.02
"""A terrain variant fits the reference if no body origin is more than this (m) below the surface under it (one
step clip lands an ankle origin 1 mm below its box top in every variant; one locomotion variant cuts 0.56 m)."""


def load_php_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = np.load(path)
    assert list(d["joint_names"]) == JOINT_NAMES, f"{path}: joint order differs from JOINT_NAMES"
    jp = d["joint_pos"]
    return jp[:, :3], jp[:, 3:7], jp[:, 7:], float(np.asarray(d["fps"]).reshape(-1)[0])


def _resample(p: torch.Tensor, quat: torch.Tensor, j: torch.Tensor, fps: float):
    """Linear (slerp for the root rotation) resampling to the control rate."""
    if abs(fps * CONTROL_DT - 1.0) < 1e-6:
        return p, quat, j
    T_src = p.shape[0]
    n = int(np.floor((T_src - 1) / fps / CONTROL_DT)) + 1
    t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * fps
    i0 = torch.clamp(t.floor().long(), max=T_src - 2)
    a = (t - i0.float()).clamp(0, 1)
    lerp = lambda x: x[i0] * (1 - a[:, None]) + x[i0 + 1] * a[:, None]  # noqa: E731
    return lerp(p), quat_slerp(quat[i0], quat[i0 + 1], a), lerp(j)


class MotionLib:
    def __init__(self, files: list[str], kin: Kinematics, device: str, root: str):
        self.device = device
        self.files = files
        self.names = [os.path.relpath(f, root)[: -len("_motion.npz")] for f in files]
        ps, qs, js, lens = [], [], [], []
        for f in files:
            p, quat, j, fps = load_php_npz(f)
            p, quat, j = _resample(*(torch.tensor(x, dtype=torch.float32) for x in (p, quat, j)), fps)
            ps.append(p), qs.append(quat), js.append(j), lens.append(len(p))
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.tensor([0] + lens[:-1], dtype=torch.long), 0).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)

        self.clip_boxes = load_clip_boxes(files).to(device)  # (C, V, K, 8)
        self.num_variants = self.clip_boxes.shape[1]
        self.goal_boxes = None  # goal-reaching terrains (attach_terrains)
        self.variant_ok = self._variants_ok(kin)

        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            nb = int(np.ceil(lens[c] / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    @torch.no_grad()
    def _variants_ok(self, kin: Kinematics) -> torch.Tensor:
        """(C, V) the variants whose terrain lies below every body origin of the reference in every frame."""
        ok = torch.ones(len(self.names), self.num_variants, dtype=torch.bool, device=self.device)
        for c in range(len(self.names)):
            boxes = self.clip_boxes[c]  # (V, K, 8)
            if not (boxes[..., 5] > 0).any():
                continue
            s, e = int(self.clip_start[c]), int(self.clip_end[c]) + 1
            P = kin.forward(*self.ref_state(torch.arange(s, e, device=self.device)), all_bodies=True)[0].reshape(-1, 3)
            h = box_heights(P[None, :, :2].expand(len(boxes), -1, -1), boxes)  # (V, T * bodies)
            ok[c] = (h - P[None, :, 2]).amax(-1) <= VARIANT_TOL
            if not ok[c].any():
                print(f"warning: every terrain variant of {self.names[c]} cuts the reference; keeping them all", flush=True)
                ok[c] = True
        return ok

    @staticmethod
    def from_dir(path: str, kin: Kinematics, device: str, pattern: str = MOTION_GLOB) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, pattern)))
        assert files, f"no motion files matching {pattern!r} in {path}"
        return MotionLib(files, kin, device, root=path)

    def attach_terrains(self, boxes: torch.Tensor):
        """Add the goal-reaching terrains ``boxes`` (G, K', 8) (:mod:`proacp.tasks.goal`): terrain index v = V + g
        is terrain g. Both box tables are padded to one box count (call before building the simulator)."""
        K = max(self.clip_boxes.shape[2], boxes.shape[1])
        pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, K - x.shape[-2]))  # noqa: E731
        self.clip_boxes = pad(self.clip_boxes)
        self.goal_boxes = pad(boxes.to(self.device))

    def boxes_of(self, clips: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(B, K, 8) boxes of terrain v: variant v of each clip, or goal-reaching terrain v - V."""
        V = self.num_variants
        own = self.clip_boxes[clips, v.clamp(max=V - 1)]
        if self.goal_boxes is None:
            return own
        return torch.where((v < V).view(-1, 1, 1), own, self.goal_boxes[(v - V).clamp(min=0)])

    def boxes(self, f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(B, K, 8) boxes of terrain v (a variant of frame f's clip, or a goal-reaching terrain)."""
        return self.boxes_of(self.clip_id[f], v)

    def sample_variants(self, clips: torch.Tensor) -> torch.Tensor:
        """A uniform valid terrain variant of each clip."""
        w = self.variant_ok[clips].float()
        return torch.multinomial(w, 1).squeeze(-1)

    def first_variants(self, clips: torch.Tensor) -> torch.Tensor:
        """Each clip's first valid terrain variant (evaluation)."""
        return self.variant_ok[clips].float().argmax(-1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
