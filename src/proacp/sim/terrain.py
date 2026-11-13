"""Box terrain of the PHP parkour clips: per-clip boxes, exact height scans and box distances in torch.

Every ``<clip>_motion.npz`` comes with ``<clip>_terrain.npy`` of shape (K, V, 10): K boxes in V variants (the
same course with each box moved or resized a little; all of them fit the motion, with rare exceptions that
:class:`proacp.clips.library.MotionLib` filters out). A box row is center (x, y, z), orientation quaternion
(w, x, y, z), a yaw rotation, and full extents (sx, sy, sz); boxes stand on the ground plane z = 0, in the
motion's world frame. Clips without obstacles have K = 0.

Boxes are stored as :data:`BOX_FIELDS` columns. The terrain height is then an exact torch function of xy and the
boxes (:func:`box_heights`), so height scans exist on imagined states as well as on simulated ones.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from proacp.sim.rotations import yaw_mat

BOX_FIELDS = 8
"""Columns of a box: center x, center y, top z, cos yaw, sin yaw, half-sizes x, y, z (all 0: padding)."""


def terrain_path(motion_file: str) -> str:
    return motion_file[: -len("_motion.npz")] + "_terrain.npy"


def load_terrain(motion_file: str) -> np.ndarray:
    """(V, K, BOX_FIELDS) boxes of every variant of one clip (K may be 0)."""
    t = np.load(terrain_path(motion_file)).astype(np.float64)
    if t.size == 0:
        return np.zeros((1, 0, BOX_FIELDS))
    K, V, _ = t.shape
    out = np.zeros((V, K, BOX_FIELDS))
    for k in range(K):
        for v in range(V):
            x, y, z, qw, qx, qy, qz, sx, sy, sz = t[k, v]
            assert abs(qx) < 1e-6 and abs(qy) < 1e-6, f"{motion_file}: box {k} is not yaw-rotated"
            yaw = 2 * math.atan2(qz, qw)
            out[v, k] = (x, y, z + sz / 2, math.cos(yaw), math.sin(yaw), sx / 2, sy / 2, sz / 2)
    return out


def load_clip_boxes(files: list[str]) -> torch.Tensor:
    """(C, V, K, BOX_FIELDS) boxes of every clip, padded to the largest box count K and variant count V (a clip
    with fewer variants repeats its first; one without boxes has only padding)."""
    per = [load_terrain(f) for f in files]
    V = max(p.shape[0] for p in per)
    K = max(1, max(p.shape[1] for p in per))
    out = torch.zeros(len(files), V, K, BOX_FIELDS)
    for c, p in enumerate(per):
        if p.shape[1] == 0:
            continue
        b = torch.tensor(p, dtype=torch.float32)
        out[c, : len(b), : b.shape[1]] = b
        out[c, len(b) :, : b.shape[1]] = b[0]
    return out


def box_heights(xy: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Terrain height (B, P) at world xy (B, P, 2) for each sample's boxes (B, K, BOX_FIELDS): the highest top
    of the boxes containing the point, 0 (ground) elsewhere. Padding boxes have zero size and never contain it.
    Piecewise constant, so it carries no gradient. One box at a time: memory O(B P), not O(B P K)."""
    x, y = xy[..., 0], xy[..., 1]
    h = torch.zeros_like(x)
    for k in range(boxes.shape[1]):
        cx, cy, top, c, s, hx, hy = boxes[:, k, None, :7].unbind(-1)  # (B, 1) each
        dx, dy = x - cx, y - cy
        inside = ((c * dx + s * dy).abs() <= hx) & ((c * dy - s * dx).abs() <= hy)  # box frame (rotate by -yaw)
        h = torch.maximum(h, torch.where(inside, top, 0.0))
    return h


def grid_offsets(x: tuple[float, float, int], y: tuple[float, float, int], device: str = "cpu") -> torch.Tensor:
    """(P, 2) points of a (start, end, count) x (start, end, count) grid in the root's yaw frame, x-major."""
    gx = torch.linspace(*x, device=device)
    gy = torch.linspace(*y, device=device)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    return torch.stack((X.flatten(), Y.flatten()), -1)


CRITIC_SCAN = ((0.0, 1.5, 31), (-0.35, 0.35, 15))
"""The height scan of the world model and the critic: a forward grid at 5 cm (privileged, never the actor's)."""


def height_scan(p: torch.Tensor, R: torch.Tensor, boxes: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """(B, P) root height above the terrain at the scan points ``offsets`` (P, 2) of the root's yaw frame.
    Differentiable in the root height; the terrain itself is piecewise constant."""
    H = yaw_mat(R)
    c, s = H[..., 0, 0:1], H[..., 1, 0:1]
    x = p[..., 0:1] + c * offsets[:, 0] - s * offsets[:, 1]
    y = p[..., 1:2] + s * offsets[:, 0] + c * offsets[:, 1]
    return p[..., 2:3] - box_heights(torch.stack((x, y), -1).detach(), boxes)


def box_sdf(x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Signed distance (B, P) from points x (B, P, 3) to one box per sample (B, BOX_FIELDS); padding: 1e3."""
    cx, cy, top, c, s, hx, hy, hz = (box[:, i : i + 1] for i in range(8))
    dx, dy = x[..., 0] - cx, x[..., 1] - cy
    q = torch.stack(((c * dx + s * dy).abs() - hx, (c * dy - s * dx).abs() - hy, (x[..., 2] - (top - hz)).abs() - hz), -1)
    outside = torch.sqrt((torch.relu(q) ** 2).sum(-1) + 1e-10)
    inside = torch.clamp(q.max(-1).values, max=0.0)
    return torch.where(hx > 0, outside + inside, 1e3)


def box_closest_point(x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Point (B, P, 3) of one box per sample (B, BOX_FIELDS) nearest to x (B, P, 3): on its surface (points
    inside go to the nearest face)."""
    cx, cy, top, c, s, hx, hy, hz = (box[:, i : i + 1] for i in range(8))
    dx, dy, dz = x[..., 0] - cx, x[..., 1] - cy, x[..., 2] - (top - hz)
    loc = torch.stack((c * dx + s * dy, c * dy - s * dx, dz), -1)
    h = torch.stack((hx, hy, hz), -1).expand_as(loc)
    p = torch.maximum(torch.minimum(loc, h), -h)
    q = loc.abs() - h
    inside = q.max(-1, keepdim=True).values < 0
    face = torch.nn.functional.one_hot(q.argmax(-1), 3).bool()
    p = torch.where(inside & face, torch.where(loc >= 0, h, -h), p)
    return torch.stack((cx + c * p[..., 0] - s * p[..., 1], cy + s * p[..., 0] + c * p[..., 1], top - hz + p[..., 2]), -1)
