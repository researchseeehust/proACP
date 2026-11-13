"""Render a processed motion set (proacp.clips.refine) with its contact labels: the processed robot solid, its source clip a
translucent ghost, the clip's boxes, and each contact as one sphere at the point where it started, which is the
tracker's target (feet red / blue, knees orange, hands green, elbows purple, pelvis yellow). One video per ``--per_video``
clips of a motion folder, climbs first; each clip in its first terrain variant.

    python tools/render_labels.py data/php_parkour_refined logs/refined_videos
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from proacp.robot import MOTION_DIR
from proacp.clips.library import load_php_npz
from render import TARGET_RGBA, scene_xml  # tools/render.py (the script's folder is on sys.path)
from proacp.sim.terrain import load_terrain

SHORT = ("Lfoot", "Rfoot", "Lknee", "Rknee", "Lhand", "Rhand", "Lelb", "Relb", "pelvis")
ORDER = ("motions-high-climb-76", "motions-low-climb-76", "motions-high-step", "motions-low-step", "motions-locomotion")


def onset_points(c: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """``pos`` (T, B, 3) with every contact of ``c`` (T, B) held at its first resolved point."""
    pos = pos.copy()
    for b in range(c.shape[1]):
        e = np.diff(np.r_[0, c[:, b].astype(np.int8), 0])
        for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            ok = np.flatnonzero(np.isfinite(pos[s:t, b]).all(-1))
            if len(ok):
                pos[s:t, b] = pos[s + ok[0], b]
    return pos


def render_video(job: tuple) -> str:
    out, motion_dir, rels, W, H, stride = job
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(50 // stride),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", out + ".tmp.mp4"], stdin=subprocess.PIPE)  # fmt: skip
    for rel in rels:
        path = os.path.join(motion_dir, rel + "_motion.npz")
        m, lab = np.load(path), np.load(os.path.join(motion_dir.rstrip("/") + "_contacts", rel + ".npz"))
        src = os.path.join(MOTION_DIR, rel + "_motion.npz")
        ref_p, ref_q, ref_j, fps = load_php_npz(src)
        jp = m["joint_pos"]
        model = mujoco.MjModel.from_xml_string(scene_xml(load_terrain(path)[0]))
        model.vis.global_.offwidth, model.vis.global_.offheight = W, H
        model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.45, 0.6
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, H, W)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation = 3.0, -18.0
        cam.azimuth = np.degrees(np.arctan2(jp[-1, 1] - jp[0, 1], jp[-1, 0] - jp[0, 0])) + 110.0  # side-on
        nq = jp.shape[1]
        c = lab["contact_info"][0, ..., 0] > 0.5
        pos, dist = onset_points(c, lab["contact_pos_w"][0]), lab["contact_dist"][0]
        for t in range(0, len(jp), stride):
            data.qpos[:nq] = jp[t]
            data.qpos[nq : nq + 3], data.qpos[nq + 3 : nq + 7], data.qpos[nq + 7 :] = ref_p[t], ref_q[t], ref_j[t]
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = (jp[t, 0], jp[t, 1], 0.6)
            renderer.update_scene(data, cam)
            scn = renderer.scene
            for b in np.flatnonzero(c[t]):
                if np.isfinite(pos[t, b]).all() and scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0]), pos[t, b].astype(np.float64),
                                        np.eye(3).reshape(-1), np.array(TARGET_RGBA[b], dtype=np.float32))  # fmt: skip
                    scn.ngeom += 1
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.text((8, 6), f"{rel}  t={t / fps:.2f}s", fill=(0, 0, 0))
            draw.text((8, 20), "contact: " + "  ".join(f"{SHORT[b]} {1000 * dist[t, b]:+.0f}mm" for b in np.flatnonzero(c[t])), fill=(0, 0, 0))
            ffmpeg.stdin.write(img.tobytes())
        renderer.close()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    os.replace(out + ".tmp.mp4", out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--per_video", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--stride", type=int, default=2, help="frames; the video stays real time")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rels = sorted(os.path.relpath(f, args.motion_dir)[: -len("_motion.npz")] for f in glob.glob(os.path.join(args.motion_dir, "*", "*_motion.npz")))
    jobs = []
    for folder in ORDER:
        fr = [r for r in rels if r.startswith(folder + "/")]
        for i in range(0, len(fr), args.per_video):
            out = os.path.join(args.out_dir, f"{folder.removeprefix('motions-')}_{i // args.per_video + 1:02d}.mp4")
            jobs.append((out, args.motion_dir, fr[i : i + args.per_video], args.width, args.height, args.stride))
    print(len(rels), "clips,", len(jobs), "videos", flush=True)
    with Pool(args.workers) as pool:
        for out in pool.imap_unordered(render_video, jobs):
            print("done", out, flush=True)


if __name__ == "__main__":
    main()
