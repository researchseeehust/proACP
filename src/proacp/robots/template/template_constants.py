"""Template of a robot for ACP: copy it, fill it in, and select it with ``PROACP_ROBOT``.

    cp src/proacp/robots/template/template_constants.py my_robot.py   # edit it
    python tools/check_robot.py --robot my_robot.py --motion_dir <your motions>
    PROACP_ROBOT=my_robot.py python -m proacp.train.stage1 --log_dir logs/my_robot_stage1

The names below are those of a humanoid called "my_robot": replace each by the link / joint of your MJCF. What every name means is
in the README (section "Your own robot"); ``proacp/robots/g1/g1_constants.py`` is the complete example. This module is not selectable as it is (``PROACP_ROBOT=template``
would load it, and ``check_robot`` would report the placeholders that do not exist in a model).
"""

from __future__ import annotations

import os
import re

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# ------------------------------------------------------------------------------------------------ files
HERE = os.path.dirname(os.path.abspath(__file__))  # a robot file of your own keeps its model next to it
MJCF_PATH = os.path.join(HERE, "my_robot.xml")
"""An MJCF with a free-floating root (a ``<freejoint>`` on ROOT_BODY), every link kept, and the visual geoms with contype =
conaffinity = 0. ``tools/urdf_to_mjcf.py`` shows how to build one from a URDF."""
DATA_DIR = os.environ.get("PROACP_DATA", os.path.join(HERE, "data"))
MOTION_DIR = os.path.join(DATA_DIR, "my_motions")
"""``<folder>/<clip>_motion.npz`` (``joint_names``, ``joint_pos`` [T, 7 + NUM_J]: root position, root quaternion wxyz, joints, and
``fps``) and ``<clip>_terrain.npy`` (the clip's boxes, in the PHP parkour format)."""
MOTION_GLOB = "*/*_motion.npz"

# ------------------------------------------------------------------------------------------------ joints and links
# the order of the motion files' ``joint_names``, of the policy's action and of every per-joint array
JOINT_NAMES = [
    "left_hip_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_joint", "right_knee_joint", "right_ankle_joint",
    "waist_joint", "left_shoulder_joint", "left_elbow_joint", "right_shoulder_joint", "right_elbow_joint",
]  # fmt: skip
NUM_J = len(JOINT_NAMES)

ROOT_BODY = "pelvis"  # the free-floating root
TORSO_BODY = "torso_link"  # its centre of mass is randomised; the goal command's torso reference
TRACKED_BODIES = [
    "pelvis", "left_knee_link", "left_foot_link", "right_knee_link", "right_foot_link", "torso_link",
    "left_elbow_link", "left_hand_link", "right_elbow_link", "right_hand_link",
]  # fmt: skip
"""Links the reward and the contact targets are measured on; the first is ROOT_BODY, and every frame link of CONTACT_BODIES is one."""

# the nine bodies whose next contact the planner predicts, in this order (the keys are fixed): name: (target frame link, the links
# whose collision geoms count as touching the terrain)
CONTACT_BODIES = {
    "left_foot": ("left_foot_link", ("left_foot_link",)),
    "right_foot": ("right_foot_link", ("right_foot_link",)),
    "left_knee": ("left_knee_link", ("left_knee_link",)),
    "right_knee": ("right_knee_link", ("right_knee_link",)),
    "left_hand": ("left_hand_link", ("left_hand_link",)),
    "right_hand": ("right_hand_link", ("right_hand_link",)),
    "left_elbow": ("left_elbow_link", ("left_elbow_link",)),
    "right_elbow": ("right_elbow_link", ("right_elbow_link",)),
    "pelvis": ("pelvis", ("pelvis",)),
}
NB = len(CONTACT_BODIES)
HAND_BODIES = ("left_hand_link", "right_hand_link")
"""Hand links that get a collision geom if the MJCF has none (see get_spec): the climbing references put the palms on the boxes."""

# ------------------------------------------------------------------------------------------------ size of the robot
STAND_HEIGHT = 0.8  # m, the root above its support when standing in your motions (goal command)
INIT_HEIGHT = 0.8  # m, the root's height at the start of an episode
TERM_HEIGHT = 0.25  # m, below this the episode ends as a fall (the lowest of your references is the guide: keep it under that)

# ------------------------------------------------------------------------------------------------ actuators
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0
JOINT_MOTORS = {  # joint name without left_/right_/_joint: (effort limit [N m], velocity limit [rad/s], rotor armature [kg m^2])
    "hip": (88.0, 32.0, 0.0102),
    "knee": (139.0, 20.0, 0.0251),
    "ankle": (50.0, 37.0, 0.0072),
    "waist": (88.0, 32.0, 0.0102),
    "shoulder": (25.0, 37.0, 0.0036),
    "elbow": (25.0, 37.0, 0.0036),
}


def _kp(arm):
    return arm * NATURAL_FREQ**2  # stiffness from the armature (BeyondMimic)


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


def motor_of(joint_name: str) -> tuple[float, float, float]:
    return JOINT_MOTORS[joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")]


def action_scale(joint_name: str) -> float:
    effort, _, arm = motor_of(joint_name)
    return 0.25 * effort / _kp(arm)  # the action's unit: a quarter of the torque limit, as a position offset


def pd_params() -> tuple[list[float], list[float], list[float], list[float]]:
    """Per joint (JOINT_NAMES order): stiffness, damping, effort limit, velocity limit."""
    kp, kd, eff, vel = [], [], [], []
    for n in JOINT_NAMES:
        e, v, arm = motor_of(n)
        kp.append(_kp(arm)), kd.append(_kd(arm)), eff.append(e), vel.append(v)
    return kp, kd, eff, vel


DEFAULT_JOINT_POS = {  # regex -> angle [rad]; the pose the action is an offset from, and the reset pose (everything else: 0)
    ".*_hip_joint": -0.3,
    ".*_knee_joint": 0.6,
    ".*_ankle_joint": -0.3,
}


def default_joint_pos_list() -> list[float]:
    out = []
    for name in JOINT_NAMES:
        val = 0.0
        for pat, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pat, name):
                val = v
        out.append(val)
    return out


# ------------------------------------------------------------------------------------------------ the entity
def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    # No self-collisions: collision geoms only touch the terrain (ground and boxes), whose geoms keep contype 1. Visual geoms
    # (contype = conaffinity = 0) stay inert, except the hands, which collide too (see HAND_BODIES).
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity or geom.parent.name in HAND_BODIES:
            geom.contype, geom.conaffinity = 0, 1
    return spec


def _actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    acts = []
    for name in JOINT_NAMES:
        effort, _, arm = motor_of(name)
        acts.append(BuiltinPositionActuatorCfg(target_names_expr=(name,), stiffness=_kp(arm), damping=_kd(arm),
                                               effort_limit=effort, armature=arm))  # fmt: skip
    return tuple(acts)


def robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, INIT_HEIGHT), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=_actuators(), soft_joint_pos_limit_factor=0.9),
    )
