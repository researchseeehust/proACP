"""The robot ACP trains, as module-level names.

``PROACP_ROBOT`` picks it (default ``g1``):

    PROACP_ROBOT=g1                        ``proacp/robots/g1/g1_constants.py``, the Unitree G1 (29 DoF)
    PROACP_ROBOT=<name>                    ``proacp/robots/<name>/<name>_constants.py``, a robot package of this repository
    PROACP_ROBOT=path/to/my_robot.py       a file outside the repo
    PROACP_ROBOT=my_package.my_robot       an importable module

Everything else in proACP imports its robot facts from here, so nothing outside ``robots/<name>/<name>_constants.py`` names a robot. A
robot module defines the names in :data:`REQUIRED` (see ``proacp/robots/template/template_constants.py`` and the README, section "Your own
robot"); a missing one is reported by name when this module is imported, and ``python tools/check_robot.py`` checks the rest.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

REQUIRED = (
  # the model
  "MJCF_PATH", "get_spec", "robot_cfg",
  "JOINT_NAMES", "NUM_J", "TRACKED_BODIES", "CONTACT_BODIES", "NB", "HAND_BODIES",
  "ROOT_BODY", "TORSO_BODY",
  # actuators and the default pose
  "pd_params", "action_scale", "DEFAULT_JOINT_POS", "default_joint_pos_list",
  # the size of the robot
  "STAND_HEIGHT", "INIT_HEIGHT", "TERM_HEIGHT",
  # the motion data
  "MOTION_DIR", "MOTION_GLOB",
)  # fmt: skip

CONTACT_ROLES = ("left_foot", "right_foot", "left_knee", "right_knee", "left_hand", "right_hand", "left_elbow", "right_elbow", "pelvis")
"""The keys of ``CONTACT_BODIES``, in this order: the planner predicts the next contact of each of these nine bodies."""


def _load(name: str):
  key = name.lower()
  if key.isidentifier() and importlib.util.find_spec(f"proacp.robots.{key}.{key}_constants") is not None:  # not a path or a dotted module
    return importlib.import_module(f"proacp.robots.{key}.{key}_constants")
  path = Path(name).expanduser()
  if name.endswith(".py") or path.is_file():
    if not path.is_file():
      raise FileNotFoundError(f"PROACP_ROBOT={name!r}: no such file")
    mod_name = f"proacp_robot_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
  try:
    return importlib.import_module(name)
  except ModuleNotFoundError as exc:
    if exc.name and name.startswith(exc.name):
      raise ValueError(f"PROACP_ROBOT={name!r}: no proacp/robots/{key}/{key}_constants.py, no such file or module") from exc
    raise


NAME = os.environ.get("PROACP_ROBOT", "g1").strip()
_module = _load(NAME)
_missing = [n for n in REQUIRED if not hasattr(_module, n)]
if _missing:
  raise AttributeError(f"the robot {NAME!r} ({_module.__name__}) does not define: {', '.join(_missing)}")
globals().update({n: getattr(_module, n) for n in REQUIRED})
