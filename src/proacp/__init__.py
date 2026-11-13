"""Attention-based contact planner (ACP) for a humanoid on box terrain (the Unitree G1, or your robot): "Humanoid Mobility via Attention-based Contacts"
(He, Wen, Zhang et al., arXiv:2610.31608) trained with SuperMimic (actor gradients through imagined rollouts of one
learned world model), not PPO.

Two networks at 50 Hz. The contact planner reads a 17 x 11 height scan around the pelvis, 5 steps of proprioception and
a goal, and gives, for each of 9 bodies (feet, knees, hands, elbows, pelvis), the terrain point of its next contact
within 1.5 s, or "no contact". The contact tracker reads 10 steps of proprioception, the 9 targets and the goal, and
outputs joint position targets; it also sees the paper's 9 x 7 local height scan. Every quantity is in the pelvis
heading frame. The world model and the critic see an exact height scan of the episode's terrain in every stage.

Self-contained on mjlab (MuJoCo-Warp); data: the PHP parkour G1 bundles (``data/php_parkour/motions-*``) or the same format
for your robot. ``PROACP_ROBOT`` picks the robot (``robot.py``).

Subpackages (each has its own docstring):
    robot.py, robots/       the active robot's names (``PROACP_ROBOT``); one package per robot: ``g1/`` (the Unitree G1),
                            ``template/`` (what a robot of your own defines)
    sim/                    ``env`` (the box-terrain simulator), ``terrain`` (boxes, height scans), ``kinematics`` (torch FK),
                            ``rotations`` (from supermimic_vibed)
    clips/                  ``library`` (the motion library), ``labels`` (per-frame contact labels), ``refine`` (clips cleaned
                            against their terrain)
    tasks/                  ``tracking`` (observation, tracking reward, termination), ``contact`` (the contact command, sphere
                            geometry of the bodies, the paper's Stage 1 reward), ``goal`` (Stage 3: procedural terrains, goal
                            reward and termination, the curriculum)
    models/                 ``networks`` (world model, policy, critic), ``planner`` (the attention planner, hindsight labels,
                            its replay buffers)
    train/                  ``stage1``, ``planner_offline``, ``stage2``, ``stage3`` (commands), ``buffer``, ``reset_sampling``
    evaluation/             ``clips`` (every clip once, with reference or planner targets), ``goals`` (goal reaching on every
                            procedural terrain type and level)

Commands (``python -m proacp.<package>.<name>``), in pipeline order:
    clips.labels            per-frame contact labels of every clip and terrain variant
    clips.refine            the clips cleaned against their terrain (no penetration, exact contacts); label them again
    train.stage1            Stage 1: the contact tracker
    train.planner_offline   Stage 2, offline: the planner from the frozen tracker
    train.stage2            Stage 2: planner and tracker co-training
    train.stage3            Stage 3: reference following and goal reaching on procedural terrains
    evaluation.clips        every clip once, with reference or planner targets
    evaluation.goals        goal reaching on every procedural terrain type and level

Tools (``python tools/<name>.py``): render, render_labels (videos of ``evaluation.clips --record`` rollouts; of a motion set with its
contact labels), check_robot (a robot definition and your motion files), test_env (FK against the simulator), urdf_to_mjcf,
and ``tools/download_data.sh``.
"""
