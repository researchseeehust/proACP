# Reference

The longer tables and explanations behind the [README](../README.md): what proACP changes, where it deviates from the contact paper,
how to read the numbers, the main training flags, the metrics, the procedural terrains and how to bring your own robot.

## What proACP changes

The baseline is the mjlab port of `supermimic_vibed`: it tracks flat-ground LAFAN1 clips with one world model and no contact signal.
proACP changes the task, the data and the networks around it. The first table lists what it adds; the second, where it deviates from
the contact paper.

| change | where | kind | what it does |
|---|---|---|---|
| Box-terrain simulator | `sim/env.py`, `sim/terrain.py` | systems | A batched mjlab simulator with per-world box slots: every clip is played on its own terrain, in one of 10 variants. |
| Contact interface | `tasks/contact.py`, `clips/labels.py` | task | The tracker's command is, for each of 9 bodies (feet, knees, hands, elbows, pelvis), the terrain point of its next contact within 1.5 s, plus a goal command and a local height scan. |
| Differentiable contact | `tasks/contact.py` (`BodySpheres`) | modelling | Touching is a sigmoid of the signed distance of each body's sphere-covered collision geoms to the boxes, so contact terms have a gradient through the world model. The paper measures contact forces, which the world model does not predict. |
| Refined motion data | `clips/refine.py` | data | A whole-trajectory IK over root pose and joints puts planted feet and palms exactly on the surface and keeps bodies out of the boxes; the published clips leave feet up to ~4 cm off the boxes and palms up to 9 cm inside them. |
| Offline planner | `models/planner.py`, `train/planner_offline.py` | algorithm | An attention planner trained on hindsight labels of the frozen tracker's own episodes: where each body's next contact actually landed. |
| Co-training | `train/stage2.py` | algorithm | Planner and tracker train together; η_obs and η_rew hand the tracker over to the planner's targets, and the ramps pause while the hindsight admission rate is low (`--hold_admit`). |
| Termination window | `tasks/tracking.py` (`--term_window`) | training / evaluation | The reference-anchored termination accepts any reference frame within ±`term_window` s, so a robot that keeps its own pace is not ended for lateness alone. |
| Procedural terrains | `tasks/goal.py`, `train/stage3.py` | data (Stage 3, untrained) | 9 box-built types × 10 levels with a success-driven curriculum, and a goal command and reward for terrains with no reference. |
| Robot interface | `robot.py`, `robots/<name>/` | engineering | `PROACP_ROBOT` selects a robot package; `tools/check_robot.py` validates one. See [Your own robot](#your-own-robot). |

| aspect | paper | here | why |
|---|---|---|---|
| tracker learning | PPO | SuperMimic: BPTT through 32 imagined steps of one world model, critic bootstrap | method choice |
| contact detection | contact forces | signed distance of a sphere approximation of each body's collision geoms to the terrain (24 spheres per convex mesh) | differentiable, available on imagined states |
| reward indicators (Stage 3) | 0/1 | sigmoids at the same thresholds | an indicator gives no gradient through the world model |
| advantage normalization per group (Stage 3) | yes | each group's imagined return divided by an EMA of its own spread | no advantages without PPO |
| stumbling term | yes | not used | needs contact forces |
| terrains | ladders, get-up, scanned meshes, … | box-built types only | box slots per simulated world |
| goal-buffer eviction | quality-based (positive work per goal distance) | FIFO per terrain type, sampling balanced across types | simplification |
| fall termination | not specified | pelvis < 0.25 m above the terrain, or tilt 1 − R_zz > 0.8 | just beyond the G1 references' extremes (0.345 m, 0.695) |
| perception | LiDAR neural mapping | exact height scan from the box geometry, Gaussian noise 3 cm (planner) / uniform 5 cm (tracker) | simulation only; no mapping noise model yet |
| domain randomization | Table S2 | friction U(0.3, 1.2), joint offsets ±0.01 rad, torso CoM offset; no pushes | a subset of Table S2 |
| motion data | retargeted with minimum-distance and contact constraints | PHP parkour clips as published, then `clips.refine` | the published clips leave feet and palms off the surface |
| hand collision | not specified | the rubber-hand mesh collides (the MJCF has no hand collision geom) | the climbs put the palms on the boxes |

Two networks run at 50 Hz. Every quantity is expressed in the pelvis heading frame (origin at the pelvis, rotated by its yaw only), so
deployment needs no global odometry.

| network | inputs | output |
|---|---|---|
| **Contact planner** (paper Table S3, attention) | 17 × 11 height scan (0.1 m cells, 1.6 × 1.0 m), 5 steps of proprioception, goal command | per body: the top-1 terrain point of its next contact within 1.5 s, and an occupancy (contact or not) |
| **Contact tracker** | 10 steps of proprioception, the 9 contact targets, goal command, 9 × 7 local height scan (0.8 × 0.6 m) | joint position targets (PD at 200 Hz) |

The goal command is the paper's bounded direction d_xy / max(‖d_xy‖, r) and a relative heading; r is 0.3 m toward the reference
0.5–1.5 s ahead and 0.5 m toward a terrain goal. The world model and the critics also see an exact height scan of the terrain in every
stage.

## How to read these numbers

- **Simulation only, exact perception.** No number comes from hardware. The planner's height scan is computed from the box geometry
  with added noise, not from a sensor pipeline; results on a real robot depend on the mapping.
- **One seed per row, no evaluation spread.** MuJoCo Warp on the GPU is not bit-deterministic, so re-evaluating one checkpoint can move a
  metric slightly; the spread has not been measured. Stage 3 trains and evaluates on the same terrain generator
  (`evaluation.goals --terrain_seed` gives unseen instances of the same types).
- **Evaluate Stage 2 with the training action noise.** Without it (`proacp.evaluation.clips` without `--train_noise`) the same checkpoints fail
  far more: 165 of 296 at 40k, and the Stage 1 tracker at 44k fails 170 instead of 7. Observation, target or scan noise alone does not
  recover them, and a noisy previous-action input alone does not either; the action noise has to reach the robot, and σ = 0.05 is not
  enough (101 failed). The world model and the critic only ever see rollouts with σ = 0.1; why the noise-free policy degrades is not
  identified yet. This is the main risk for a deployment, where the noise is not added.
- **Later Stage 1 checkpoints evaluate worse.** On the Stage 1 evaluation, later checkpoints of both runs are worse (8,000: 21 failed)
  although the training statistics keep improving; the cause is not identified, and the reported checkpoint (4000) was picked on that
  evaluation, which favours it.
- **Not comparable one to one with the paper.** The robot data (PHP parkour clips), the learner (SuperMimic, no PPO baseline here) and
  the terrains (boxes only) differ; see the second table under [What proACP changes](#what-proacp-changes).
- **The logs are not in the repository.** Unlike a repository that ships its per-run logs, the tables here cannot be regenerated from
  files; `scripts/reproduce_main.sh` reruns the training and the evaluations (about 21 h on one GPU).

## Training flags

Main flags:

| command | flag | default | meaning |
|---|---|---|---|
| all | `--num_envs` | 4096 | parallel simulated robots |
| all | `--horizon` | 32 | imagination horizon |
| all | `--seed` | 0 | random seed |
| `stage1` | `--iterations` | 60000 | |
| `stage1` | `--blind` | off | drop the tracker's 9 × 7 height scan |
| `stage1` | `--reward` | `paper` | `paper` (Stage 1 reward) or `supermimic` (tracking reward + contact labels) |
| `stage1` | `--backward_curriculum` | on | the paper's start-phase curriculum |
| `planner_offline` | `--steps` | 20000 | simulator steps |
| `stage2` | `--u_obs`, `--u_rew`, `--n_ann` | 2000, 12000, 20000 | η_obs / η_rew ramps (Eq. 3), in iterations |
| `stage2` | `--hold_admit` | 0.6 | pause the ramps while the hindsight admission rate drops below this share |
| `stage2`, `stage3` | `--term_window` | 0 | s: tolerance of the reference termination to being early or late |
| `stage3` | `--goal_frac` | 0.5 | share of goal-reaching environments and imagination samples |
| `stage3` | `--critic_warmup` | 500 | iterations before goal samples enter the actor loss |
| `stage3` | `--terrain_instances`, `--terrain_seed` | 16, 0 | the procedural terrain bank |
| `stage3` | `--start_level` | 0 | goal environments start at a uniform level up to this |

## Metrics

Protocol: no randomization, no observation noise, deterministic policy, friction 1.0, each clip on its first valid terrain variant.
Results go to a json next to the checkpoint.

- *Normalized return*: the SuperMimic tracking reward summed over the clip, divided by its maximum. A clip fails if the training
  termination triggers.
- *Touch precision / recall*: per body, frames touching the terrain (sphere approximation within 1 cm) against the reference labels.
- *Placement error*: per body, the distance from where each new contact lands to the nearest target commanded over the last 5 steps.
- *Planner F1 / top-1 error*: occupancy F1, and the distance from the top-1 point to the hindsight label, on held-out clips.
- *Goal success*: within 0.5 m horizontally and 0.1 m vertically of the goal, with no fall before the episode ends (15 s).

## Procedural terrains (Stage 3)

Not trained yet. `train.stage3` and `evaluation.goals` are in the repository; `evaluation.goals` reports, per terrain type and level, the
success rate, the fall rate and the time to the goal. The procedural terrains are built from boxes: 9 types × 10 difficulty levels × N
random instances, each rotated by a random yaw.

| type | course | difficulty (level 0 → 9) |
|---|---|---|
| stairs_up / walled_stairs | 4–6 steps up to a landing (walls on both sides) | step height 5 → 20 cm |
| stairs_down | start on a landing, 4–6 steps down | step height 5 → 20 cm |
| platform | walk-up platform of random orientation | 10 → 40 cm |
| climb | climbable obstacle | 40 → 85 cm |
| gap | start platform, gap, goal platform | gap 10 → 60 cm |
| stepping_stones | 6 stones between two platforms | stones 45 → 25 cm, gaps 5 → 35 cm |
| balance_beam | beam between two platforms | width 50 → 20 cm |
| pallets | narrow slats with gaps | slats 20 → 10 cm, gaps 5 → 25 cm |

## Your own robot

A robot is **one file** that defines its model, the order of its links and joints, its PD gains and the links that play a role.
`PROACP_ROBOT` selects it, and nothing else in the code names a robot: `robot.py` re-exports the names of the chosen robot module, and
the simulator, rewards, planner, terrains and renderers import them from there. `src/proacp/robots/g1/g1_constants.py` is the complete
example, `src/proacp/robots/template/template_constants.py` the skeleton.

```bash
PROACP_ROBOT=g1                        # Unitree G1, 29 DoF (default): proacp/robots/g1/g1_constants.py
PROACP_ROBOT=my_robot                  # proacp/robots/my_robot/my_robot_constants.py in this repository
PROACP_ROBOT=path/to/my_robot.py       # a file anywhere
PROACP_ROBOT=my_package.my_robot       # an importable module
```

**What you need.** An MJCF of a humanoid with hinge joints only, a free-floating root and collision geoms on the nine contact bodies;
motion clips of that robot in the PHP parkour format, each with the boxes of its terrain; and a GPU as for the G1. The published bundles
are for the G1. **Retargeting human parkour motion to another robot is outside proACP**; the quality of the clips (feet that float
above or sink into the boxes) decides how well the tracker can follow them, and `clips.refine` cleans only what a whole-trajectory IK
on your model can fix.

**The model.** A free joint on the root body and every link kept (no merged fixed links), since the tracked bodies are links of the
tree (`tools/urdf_to_mjcf.py` turns a URDF into such an MJCF). Hinge joints only, each with a range. Visual geoms with
`contype = conaffinity = 0`; collision geoms (sphere, capsule or convex mesh) on the links of the nine contact bodies, which
`get_spec()` gives `contype 0 / conaffinity 1`: they touch the terrain and nothing else, so there are no self-collisions. The hands need
a collision geom, since the climbing references put the palms on the box tops; if your MJCF has none, list the hand links in
`HAND_BODIES` and `get_spec()` makes their mesh collide. The robot is simulated at 200 Hz (4 substeps of 5 ms) with a PD law per joint
and controlled at 50 Hz.

**The robot file.** `cp src/proacp/robots/template/template_constants.py my_robot.py`, then fill in:

| names | what |
|---|---|
| `MJCF_PATH`, `get_spec()`, `robot_cfg()` | the model, as a `mujoco.MjSpec` and an mjlab `EntityCfg` |
| `JOINT_NAMES`, `NUM_J` | the hinge joints, in the order of the motion files' `joint_names` and of the action |
| `ROOT_BODY`, `TORSO_BODY` | the free root; the torso (its centre of mass is randomised, the goal command refers to it) |
| `TRACKED_BODIES` | links the rewards and the contact targets are measured on; the first is `ROOT_BODY` |
| `CONTACT_BODIES`, `NB` | the nine bodies whose next contact the planner predicts: `left_foot`, `right_foot`, `left_knee`, `right_knee`, `left_hand`, `right_hand`, `left_elbow`, `right_elbow`, `pelvis`, in this order. Each maps to `(frame link, links whose collision geoms count)`; the frame link is in `TRACKED_BODIES` |
| `HAND_BODIES` | hand links that get a collision geom if the MJCF has none |
| `pd_params()`, `action_scale(joint)` | per joint: stiffness, damping, effort and velocity limit; the action's unit (the G1 derives them from each motor's armature) |
| `DEFAULT_JOINT_POS`, `default_joint_pos_list()` | the default pose: the action is an offset from it, and episodes start in it |
| `STAND_HEIGHT`, `INIT_HEIGHT`, `TERM_HEIGHT` | root height (m) standing in your motions, at the start of an episode, and below which an episode ends as a fall |
| `MOTION_DIR`, `MOTION_GLOB` | where the motion files are |

`TERM_HEIGHT` must stay below the lowest root height of your references (the G1's go down to 0.345 m while climbing; the default is
0.25 m); a higher value ends every clip as a fall.

**The motion data.** `<MOTION_DIR>/<group>/<clip>_motion.npz` holds `joint_names` [NUM_J], `joint_pos` [T, 7 + NUM_J] (root x y z, root
quaternion wxyz, joints) and `fps`; `<clip>_terrain.npy` the boxes of the clip's terrain in every variant (the PHP format).
`joint_names` must equal `JOINT_NAMES`: a file in another joint order is rejected. Clips are resampled to 50 Hz.

**Check, then train.**

```bash
python tools/check_robot.py --robot my_robot.py --motion_dir data/my_motions    # the declaration, the model, your motion files
PROACP_ROBOT=my_robot.py python tools/test_env.py --motion_dir data/my_motions  # torch FK vs MuJoCo, files vs the robot
PROACP_ROBOT=my_robot.py PROACP_DATA=data scripts/prepare_data.sh               # then scripts/train.sh stage1 ...
```

`check_robot` checks the names, lengths, the nine contact roles and heights; that the MJCF compiles, the root is free, every named body
and joint exists, the joints are hinges with a range and every contact body has a collision geom; that the torch FK and the contact
spheres build; and your motion files (`joint_names`, width of `joint_pos`, terrain files, the lowest root height against
`TERM_HEIGHT`).

**What is still tuned for the G1.** These are not robot names yet and may need your attention for a robot of another size: the
terrain difficulty of `tasks/goal.py` (step heights 5 to 20 cm, climbs 40 to 85 cm, gaps 10 to 60 cm); the goal speeds (0.8 m/s on stairs,
2.0 elsewhere) and the reward constants of `tasks/tracking.py` and `tasks/contact.py`; the tilt test of the fall termination (`1 − R_zz > 0.8`); the
contact geometry (24 spheres per convex mesh, touch within 1 cm); and the domain randomisation of `sim/env.py`. **Only the G1 has been
trained**: a robot loaded from your own file is checked, but no second robot has been trained.
