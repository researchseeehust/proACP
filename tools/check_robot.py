"""Check a robot definition before training with it.

    python tools/check_robot.py [--robot g1 | name | path/to/my_robot.py] [--motion_dir DIR]

1. The declaration: names, lengths, the nine contact roles, frame links, heights, actuator arrays.
2. The model it builds: the MJCF compiles, the root is free, every named body and joint exists, the joints are hinges with a range,
   every contact body has a collision geom, and ``Kinematics`` / ``BodySpheres`` (the torch FK and the contact spheres) build.
3. With ``--motion_dir`` (default: the robot's ``MOTION_DIR``, when it exists): the motion files carry the robot's
   ``joint_names``, ``joint_pos`` has ``7 + NUM_J`` columns, each clip has its terrain file, and the lowest root height of the
   clips is above ``TERM_HEIGHT``.

Errors stop ACP from using the robot; warnings are worth reading. Exit status 1 when anything failed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--robot", default=None, help="g1, a name (proacp/robots/<name>/<name>_constants.py), a .py file or a module (default: $PROACP_ROBOT, else g1)")
  p.add_argument("--motion_dir", default=None, help="folder of <group>/<clip>_motion.npz (default: the robot's MOTION_DIR)")
  p.add_argument("--clips", type=int, default=8, help="how many motion files to read")
  a = p.parse_args()
  if a.robot:
    os.environ["PROACP_ROBOT"] = a.robot  # before anything imports proacp.robot

  errors: list[str] = []
  warns: list[str] = []

  from proacp import robot as R

  print(f"robot {R.NAME!r}: {R.NUM_J} joints, root {R.ROOT_BODY!r}, {len(R.TRACKED_BODIES)} tracked bodies, {R.NB} contact bodies")

  # ---------------------------------------------------------------------------------- 1. the declaration
  names = list(R.JOINT_NAMES)
  if len(set(names)) != len(names):
    errors.append(f"duplicate joints: {sorted({n for n in names if names.count(n) > 1})}")
  if R.NUM_J != len(names):
    errors.append(f"NUM_J {R.NUM_J} != len(JOINT_NAMES) {len(names)}")
  if list(R.CONTACT_BODIES) != list(R.CONTACT_ROLES):
    errors.append(f"CONTACT_BODIES keys must be {list(R.CONTACT_ROLES)}, in this order; they are {list(R.CONTACT_BODIES)}")
  if R.TRACKED_BODIES[0] != R.ROOT_BODY:
    errors.append(f"TRACKED_BODIES must start with ROOT_BODY {R.ROOT_BODY!r}, it starts with {R.TRACKED_BODIES[0]!r}")
  if R.TORSO_BODY not in R.TRACKED_BODIES:
    errors.append(f"TORSO_BODY {R.TORSO_BODY!r} is not in TRACKED_BODIES")
  for role, (frame, links) in R.CONTACT_BODIES.items():
    if frame not in R.TRACKED_BODIES:
      errors.append(f"CONTACT_BODIES[{role!r}]: the frame link {frame!r} is not in TRACKED_BODIES")
    if not links:
      errors.append(f"CONTACT_BODIES[{role!r}]: no links whose collision geoms count")
  if len(R.HAND_BODIES) != 2:
    errors.append("HAND_BODIES takes two links (left, right)")
  try:
    kp, kd, eff, vel = R.pd_params()
    for what, arr in (("stiffness", kp), ("damping", kd), ("effort limit", eff), ("velocity limit", vel)):
      if len(arr) != R.NUM_J:
        errors.append(f"pd_params(): {len(arr)} {what} values for {R.NUM_J} joints")
      elif min(arr) <= 0:
        errors.append(f"pd_params(): {what} must be positive (min {min(arr)})")
  except Exception as exc:  # noqa: BLE001
    errors.append(f"pd_params() fails ({exc!r}): is there a motor for every joint?")
  try:
    if len(R.default_joint_pos_list()) != R.NUM_J:
      errors.append(f"default_joint_pos_list() has {len(R.default_joint_pos_list())} values for {R.NUM_J} joints")
  except Exception as exc:  # noqa: BLE001
    errors.append(f"default_joint_pos_list() fails: {exc!r}")
  for pat in R.DEFAULT_JOINT_POS:
    if not any(re.fullmatch(pat, n) for n in names):
      warns.append(f"DEFAULT_JOINT_POS: {pat!r} matches no joint")
  try:
    scales = [R.action_scale(n) for n in names]
    if min(scales) <= 0:
      errors.append("action_scale() must be positive for every joint")
  except Exception as exc:  # noqa: BLE001
    errors.append(f"action_scale() fails: {exc!r}")
  if not 0.0 < R.TERM_HEIGHT < R.INIT_HEIGHT <= 1.5 or not R.TERM_HEIGHT < R.STAND_HEIGHT <= 1.5:
    errors.append(f"heights must satisfy 0 < TERM_HEIGHT {R.TERM_HEIGHT} < INIT_HEIGHT {R.INIT_HEIGHT} and < STAND_HEIGHT {R.STAND_HEIGHT}")

  # ---------------------------------------------------------------------------------- 2. the model
  if not errors:
    if not os.path.isfile(R.MJCF_PATH):
      errors.append(f"MJCF_PATH {R.MJCF_PATH} does not exist (G1: run tools/download_data.sh)")
    else:
      import mujoco

      try:
        m = R.get_spec().compile()
      except Exception as exc:  # noqa: BLE001
        errors.append(f"get_spec() does not compile: {exc}")
        m = None
      if m is not None:
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        free = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1 or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[free[0]])) != R.ROOT_BODY:
          errors.append(f"the model needs exactly one free joint, on ROOT_BODY {R.ROOT_BODY!r} (it has {len(free)})")
        wanted = set(R.TRACKED_BODIES) | set(R.HAND_BODIES) | {R.ROOT_BODY, R.TORSO_BODY}
        for _, links in R.CONTACT_BODIES.values():
          wanted |= set(links)
        for b in sorted(wanted):
          if body(b) < 0:
            errors.append(f"body {b!r} is not in the model")
        hinges = {m.joint(i).name: i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
        for n in names:
          if n not in hinges:
            errors.append(f"joint {n!r} is not a hinge joint of the model")
          elif not (m.jnt_limited[hinges[n]] and m.jnt_range[hinges[n]][1] > m.jnt_range[hinges[n]][0]):
            warns.append(f"joint {n!r} has no range: the joint-limit terms see an unbounded joint")
        extra = sorted(set(hinges) - set(names))
        if extra:
          errors.append(f"hinge joints of the model missing from JOINT_NAMES: {extra[:6]}{' ...' if len(extra) > 6 else ''}")
        for role, (_, links) in R.CONTACT_BODIES.items():
          ids = [body(b) for b in links if body(b) >= 0]
          geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_conaffinity[g]]
          if ids and not geoms:
            errors.append(f"contact body {role!r}: no collision geom on {list(links)} (get_spec() must give it contype 0 / conaffinity 1)")
        if not errors:
          try:
            import torch

            from proacp.sim.kinematics import Kinematics

            kin = Kinematics("cpu")
            ok = bool(torch.isfinite(kin.q_lo).all() and torch.isfinite(kin.q_hi).all())
            print(f"  model: {m.nbody - 1} bodies, {m.njnt} joints, {m.ngeom} geoms; Kinematics: {len(kin.body_names)} moving bodies")
            if not ok:
              errors.append("Kinematics: the joint limits are not finite")
          except Exception as exc:  # noqa: BLE001
            errors.append(f"Kinematics() fails: {exc!r}")
          try:
            from proacp.tasks.contact import BodySpheres

            bs = BodySpheres()
            print(f"  contact spheres: {len(bs.pts)} over {R.NB} bodies")
            for role, (lo, hi) in zip(R.CONTACT_BODIES, bs.bounds):
              if hi == lo:
                errors.append(f"contact body {role!r}: BodySpheres found no collision geom (sphere, capsule or mesh) on its links")
          except Exception as exc:  # noqa: BLE001
            errors.append(f"BodySpheres() fails: {exc!r}")

  # ---------------------------------------------------------------------------------- 3. the motion data
  motion_dir = a.motion_dir or R.MOTION_DIR
  files = sorted(glob.glob(os.path.join(motion_dir, R.MOTION_GLOB)))
  if not files:
    print(f"  motion data: none in {motion_dir} (checked only the robot; pass --motion_dir to check your clips)")
  elif not errors:
    import numpy as np

    step = max(len(files) // max(a.clips, 1), 1)
    picked = files[::step][: a.clips]
    low = np.inf
    for f in picked:
      d = np.load(f)
      if list(d["joint_names"]) != names:
        errors.append(f"{os.path.relpath(f, motion_dir)}: joint_names differ from JOINT_NAMES (the order of the file is the robot's)")
        break
      jp = d["joint_pos"]
      if jp.shape[1] != 7 + R.NUM_J:
        errors.append(f"{os.path.relpath(f, motion_dir)}: joint_pos has {jp.shape[1]} columns, expected 7 + {R.NUM_J}")
        break
      if not os.path.isfile(f.replace("_motion.npz", "_terrain.npy")):
        errors.append(f"{os.path.relpath(f, motion_dir)}: no <clip>_terrain.npy next to it")
      low = min(low, float(jp[:, 2].min()))
    if not errors:
      print(f"  motion data: {len(files)} clips in {motion_dir}; {len(picked)} read, lowest root height {low:.3f} m")
      if low < R.TERM_HEIGHT + 0.02:
        warns.append(f"the lowest root height of the clips ({low:.3f} m) is close to TERM_HEIGHT {R.TERM_HEIGHT}: the references would end as falls")

  for w in warns:
    print(f"  [warn]  {w}")
  for e in errors:
    print(f"  [ERROR] {e}")
  print("OK" if not errors else "FAILED")
  sys.exit(1 if errors else 0)


if __name__ == "__main__":
  main()
