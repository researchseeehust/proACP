"""Render recorded evaluations (proacp.evaluation.clips --record) with MuJoCo (EGL): the simulated robot solid, the
reference a translucent ghost, the clip's boxes, and the commanded contact targets as colored spheres (feet red / blue,
knees orange, hands green, elbows purple, pelvis yellow). Each clip plays until its rollout ends, plus ``--tail`` s.

    python tools/render.py traj.npz out.mp4 --clips motions-low-climb-76/forward_walk0.6s_step0.7s
    python tools/render.py traj.npz out.mp4 --per_folder 1      # the first clip of every motion folder
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from proacp.robot import MJCF_PATH, ROOT_BODY

TARGET_RGBA = [(0.9, 0.2, 0.2, 0.9), (0.2, 0.4, 0.95, 0.9), (1.0, 0.55, 0.1, 0.9), (1.0, 0.55, 0.1, 0.9), (0.2, 0.8, 0.3, 0.9),
               (0.2, 0.8, 0.3, 0.9), (0.6, 0.3, 0.8, 0.9), (0.6, 0.3, 0.8, 0.9), (0.95, 0.85, 0.2, 0.9)]  # fmt: skip


def scene_xml(boxes: np.ndarray) -> str:
    """The G1, a ghost copy (prefix ``ref_``), a floor, a light and ``boxes`` (K, 8)."""
    xml = open(MJCF_PATH).read()
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="' + os.path.dirname(MJCF_PATH) + '/"', xml)
    body = xml[xml.index(f'<body name="{ROOT_BODY}"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r' (rgba|contype|conaffinity|group|material)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" rgba="0.35 0.65 1.0 0.25" ')
    extra = ['<light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>',
             '<geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>']  # fmt: skip
    for k, (cx, cy, top, c, s, hx, hy, hz) in enumerate(boxes):
        if hx > 0:
            yaw = np.arctan2(s, c)
            extra.append(f'<geom name="box{k}" type="box" size="{hx} {hy} {hz}" pos="{cx} {cy} {top - hz}" '
                         f'quat="{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}" rgba="0.62 0.55 0.45 1" contype="0" conaffinity="0"/>')  # fmt: skip
    xml = xml.replace("</worldbody>", ghost + "\n".join(extra) + "</worldbody>")
    assets = ('<texture name="sky" type="skybox" builtin="gradient" rgb1="0.85 0.9 0.97" rgb2="0.55 0.65 0.8" width="256" height="256"/>'
              '<texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>'
              '<material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>')  # fmt: skip
    if "<asset>" in xml:
        xml = xml.replace("<asset>", "<asset>" + assets, 1)
    else:
        xml = xml.replace("<worldbody>", "<asset>" + assets + "</asset><worldbody>", 1)
    return xml


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None, help="clip names (default: --per_folder)")
    ap.add_argument("--per_folder", type=int, default=1, help="without --clips: this many clips of every motion folder")
    ap.add_argument("--seconds", type=float, default=None, help="cap per clip")
    ap.add_argument("--tail", type=float, default=1.0, help="s shown after the rollout ends")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    d = np.load(args.traj)
    names = list(d["names"])
    if args.clips:
        clips = [names.index(c) for c in args.clips]
    else:
        folders = sorted({n.split("/")[0] for n in names})
        clips = [i for g in folders for i in [j for j, n in enumerate(names) if n.startswith(g + "/")][: args.per_folder]]
    alive, f = d["alive"], d["f"]  # (T, C)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
                               "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
                              stdin=subprocess.PIPE)  # fmt: skip
    for c in clips:
        m = mujoco.MjModel.from_xml_string(scene_xml(d["boxes"][c]))
        m.vis.global_.offwidth, m.vis.global_.offheight = args.width, args.height
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = 0.45, 0.6
        data = mujoco.MjData(m)
        nq = 7 + d["q"].shape[-1]
        r = mujoco.Renderer(m, args.height, args.width)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 3.2, -15.0, 135.0
        T_alive = int(alive[:, c].sum())
        T = min(T_alive + int(args.tail * args.fps), len(alive))
        if args.seconds:
            T = min(T, int(args.seconds * args.fps))
        failed = T_alive < int(d["clip_len"][c]) - 1
        for t in range(T):
            fr = int(f[t, c])
            data.qpos[:3], data.qpos[3:7], data.qpos[7:nq] = d["root_p"][t, c], d["root_quat"][t, c], d["q"][t, c]
            data.qpos[nq : nq + 3], data.qpos[nq + 3 : nq + 7] = d["ref_root_p"][fr], mat_to_quat(d["ref_root_R"][fr])
            data.qpos[nq + 7 :] = d["ref_q"][fr]
            mujoco.mj_kinematics(m, data)
            cam.lookat[:] = (d["root_p"][t, c][0], d["root_p"][t, c][1], 0.6)
            r.update_scene(data, cam)
            scn = r.scene
            for b in np.flatnonzero(d["tgt_valid"][t, c]):
                if scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.035, 0, 0]), d["tgt_w"][t, c, b].astype(np.float64),
                                        np.eye(3).reshape(-1), np.array(TARGET_RGBA[b], dtype=np.float32))  # fmt: skip
                    scn.ngeom += 1
            img = Image.fromarray(r.render())
            ImageDraw.Draw(img).text((10, 8), f"{names[c]}  t={t / args.fps:.2f}s" + ("  FAILED" if failed and t >= T_alive else ""), fill=(0, 0, 0))
            ffmpeg.stdin.write(img.tobytes())
        r.close()
        print(f"{names[c]}: {T} frames, {'failed at %.2f s' % (T_alive / args.fps) if failed else 'completed'}", flush=True)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
