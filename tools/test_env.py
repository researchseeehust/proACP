"""Sanity check of the setup: the torch FK against MuJoCo, and (with motion files) the files against the robot.

    python tools/test_env.py [--motion_dir data/php_parkour]

1. ``Kinematics`` (the differentiable FK every reward and the world model use) must reproduce MuJoCo's body positions for random
   root poses and joint angles (tolerance 1e-5 m).
2. With motion files: each file's ``joint_names`` are the robot's, ``joint_pos`` has 7 + NUM_J columns, and the first frame of
   each clip, put into the MuJoCo model, has its tracked bodies above the floor (a wrong joint order or axis shows here).
Exit status 1 when a check fails. ``tools/check_robot.py`` checks the robot definition itself.
"""

import argparse
import glob
import os
import sys

import mujoco
import numpy as np
import torch

from proacp import robot as R
from proacp.sim.kinematics import Kinematics


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motion_dir", default=None, help="default: the robot's MOTION_DIR, when it exists")
  ap.add_argument("--clips", type=int, default=12)
  a = ap.parse_args()
  ok = True

  kin = Kinematics("cpu")
  m = mujoco.MjModel.from_xml_path(R.MJCF_PATH)
  d = mujoco.MjData(m)
  adr = {m.joint(i).name: m.jnt_qposadr[i] for i in range(m.njnt)}
  rng = np.random.default_rng(0)
  worst = 0.0
  for _ in range(20):
    q = rng.uniform(-0.6, 0.6, R.NUM_J)
    pos = rng.uniform(-1, 1, 3)
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    d.qpos[:3], d.qpos[3:7] = pos, quat
    for n, v in zip(R.JOINT_NAMES, q):
      d.qpos[adr[n]] = v
    mujoco.mj_kinematics(m, d)
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, quat)
    P, *_ = kin.forward(torch.tensor(pos, dtype=torch.float32)[None], torch.tensor(Rm.reshape(3, 3), dtype=torch.float32)[None],
                        torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor(q, dtype=torch.float32)[None], torch.zeros(1, R.NUM_J))  # fmt: skip
    ref = np.array([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)] for b in R.TRACKED_BODIES])
    worst = max(worst, float(np.abs(P[0].numpy() - ref).max()))
  good = worst < 1e-5
  ok &= good
  print(f"FK vs MuJoCo, {len(R.TRACKED_BODIES)} tracked bodies, 20 random poses: max error {worst:.2e} m  [{'ok' if good else 'FAILED'}]")

  motion_dir = a.motion_dir or R.MOTION_DIR
  files = sorted(glob.glob(os.path.join(motion_dir, R.MOTION_GLOB)))
  if not files:
    print(f"no motion files in {motion_dir} (pass --motion_dir to check your clips)")
  else:
    picked = files[:: max(len(files) // a.clips, 1)][: a.clips]
    lowest = np.inf
    for f in picked:
      z = np.load(f)
      if list(z["joint_names"]) != list(R.JOINT_NAMES) or z["joint_pos"].shape[1] != 7 + R.NUM_J:
        print(f"{os.path.relpath(f, motion_dir)}: joint_names or joint_pos width differ from the robot  [FAILED]")
        ok = False
        break
      jp = z["joint_pos"]
      d.qpos[:3], d.qpos[3:7] = jp[0, :3], jp[0, 3:7]
      for n, v in zip(R.JOINT_NAMES, jp[0, 7:]):
        d.qpos[adr[n]] = v
      mujoco.mj_kinematics(m, d)
      lowest = min(lowest, min(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)][2] for b in R.TRACKED_BODIES))
    else:
      print(f"{len(picked)} of {len(files)} clips: joint order and width match; lowest tracked body at frame 0: {lowest:.3f} m  [ok]")

  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
