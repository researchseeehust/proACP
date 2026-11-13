"""Unitree G1 (29 DoF) for the contact pipeline on mjlab (MuJoCo-Warp), BeyondMimic's actuator model.

The default robot (``PROACP_ROBOT=g1``); ``robot.py`` re-exports its names, and a robot of your own is a file like this one
(``proacp/robots/template/template_constants.py``, the README section "Your own robot").

Same robot as supermimic_vibed's ``supermimic/robot.py`` (implicit PD at a 10 Hz natural frequency, damping ratio 2 on the
reflected armature, action scale 0.25 * effort_limit / stiffness), built as an mjlab entity from the
MJCF instead of the Isaac Lab USD. Collision geoms touch the terrain only (ground and boxes); there are
no self-collisions.
"""

from __future__ import annotations

import os
import re

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA_DIR = os.environ.get("PROACP_DATA", os.path.join(REPO, "data"))  # G1 description, motion bundles, contact labels
MJCF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof.xml")
MOTION_DIR = os.path.join(DATA_DIR, "php_parkour")
"""PHP parkour bundles (``motions-*/<clip>_motion.npz`` + ``<clip>_terrain.npy``), e.g. a symlink to their folder."""
MOTION_GLOB = "*/*_motion.npz"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _kp(arm):
    return arm * NATURAL_FREQ**2


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


# Unitree SDK order (also the order of the motion files' joint_names)
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip
NUM_J = len(JOINT_NAMES)

# Roles of the links (every one is a body of the MJCF)
ROOT_BODY = "pelvis"  # the free-floating root; the first of TRACKED_BODIES
TORSO_BODY = "torso_link"  # its centre of mass is randomised; the goal command's torso reference
# Heights (m) of the root above its support: standing in the clips, at the start of an episode, and below which an episode
# ends as a fall (just beyond the references' lowest, 0.345 m while climbing).
STAND_HEIGHT = 0.78
INIT_HEIGHT = 0.76
TERM_HEIGHT = 0.25

# BeyondMimic's tracked bodies; the anchor/root is the pelvis. Every contact body's frame link is one of them.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

CONTACT_BODIES = {  # name: (target frame link, links whose collision geoms count)
    "left_foot": ("left_ankle_roll_link", ("left_ankle_roll_link",)),  # the four sole spheres
    "right_foot": ("right_ankle_roll_link", ("right_ankle_roll_link",)),
    "left_knee": ("left_knee_link", ("left_knee_link",)),  # knee and shin hull
    "right_knee": ("right_knee_link", ("right_knee_link",)),
    "left_hand": ("left_wrist_yaw_link", ("left_wrist_yaw_link", "left_rubber_hand")),  # wrist hull + hand (see get_spec)
    "right_hand": ("right_wrist_yaw_link", ("right_wrist_yaw_link", "right_rubber_hand")),
    "left_elbow": ("left_elbow_link", ("left_elbow_link",)),  # elbow and forearm hull
    "right_elbow": ("right_elbow_link", ("right_elbow_link",)),
    "pelvis": ("pelvis", ("pelvis", "pelvis_contour_link")),
}
NB = len(CONTACT_BODIES)

# Per-joint (effort limit, velocity limit, armature) by motor type.
_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}


def motor_of(joint_name: str) -> tuple[float, float, float]:
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return JOINT_MOTORS[key]


def action_scale(joint_name: str) -> float:
    effort, _, arm = motor_of(joint_name)
    return 0.25 * effort / _kp(arm)


def pd_params() -> tuple[list[float], list[float], list[float], list[float]]:
    """Per joint (JOINT_NAMES order): stiffness, damping, effort limit, velocity limit."""
    kp, kd, eff, vel = [], [], [], []
    for n in JOINT_NAMES:
        e, v, arm = motor_of(n)
        kp.append(_kp(arm)), kd.append(_kd(arm)), eff.append(e), vel.append(v)
    return kp, kd, eff, vel


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
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


HAND_BODIES = ("left_rubber_hand", "right_rubber_hand")


def get_spec() -> mujoco.MjSpec:
    # MJCF generated by tools/urdf_to_mjcf.py (free-floating pelvis, URDF visuals kept for rendering).
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    # No self-collisions: collision geoms (convex mesh hulls and foot spheres) only touch the terrain, whose
    # geoms keep contype 1. Visual geoms (contype = conaffinity = 0) stay inert, except the hands: the MJCF gives
    # them no collision geom, yet the climbing references put the palms on the box tops (within 5 mm in 17-20 % of
    # their frames), so the hand mesh collides too (its hull; density 0, so masses and inertias are unchanged).
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity or geom.parent.name in HAND_BODIES:
            geom.contype, geom.conaffinity = 0, 1
    return spec


def _actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    # The velocity limits of JOINT_MOTORS are not simulated: MuJoCo has no joint velocity limit.
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
