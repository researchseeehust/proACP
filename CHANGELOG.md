# Changelog

## 0.0.0 (2026-11-13)

First release.

- **Contact tracker and contact planner for the Unitree G1** on mjlab / MuJoCo Warp, trained with SuperMimic's differentiable world model.
  Stages 1 and 2 are trained and evaluated on the 296 PHP parkour clips (one seed; see `results/`). Stage 3 (goal reaching on procedural
  terrains) is implemented and not trained.
- **Package `proacp`** in `src/proacp/`, split by role: `sim/` (simulator, terrains, FK, rotations), `clips/` (motion library, contact
  labels, refinement), `tasks/` (tracking MDP, contact interface, goal reaching), `models/` (networks, planner), `train/` (stages, replay
  buffer, start-bin sampling), `evaluation/`, `robots/`. The commands are modules of those packages: `python -m proacp.clips.labels`,
  `clips.refine`, `train.stage1`, `train.planner_offline`, `train.stage2`, `train.stage3`, `evaluation.clips`, `evaluation.goals`.
- **Pluggable robots.** `PROACP_ROBOT` selects `proacp/robots/<name>/<name>_constants.py` (the G1 is `proacp/robots/g1/`), a `.py` file or a
  module; `tools/check_robot.py` validates a robot definition and `src/proacp/robots/template/` is the skeleton of one.
- **Data and tools.** `tools/download_data.sh` downloads the G1 description and the five PHP parkour bundles, each checked against its
  sha256; `tools/test_env.py` checks the torch FK against MuJoCo (2e-7 m) and the motion files against the robot; `tools/render.py` and
  `tools/render_labels.py` render rollouts and contact labels.
- **Scripts.** `scripts/prepare_data.sh` (labels, refinement), `scripts/train.sh` (one training command with its log folder) and
  `scripts/reproduce_main.sh` (the reported runs and the two evaluations); `experiments/runs.txt` lists the reported runs.
- **`uv.lock`** pins torch 2.11.0 (CUDA 12.8), warp-lang 1.16.0 and `mjviser<0.0.17`. torch 2.9.0 (the reported runs) fails in Triton 3.5.0 on
  an H100, and `mjviser` 0.0.17 pulls `mujoco-core`, which overwrites the files of `mujoco` 3.11.0.
- **Licenses.** `LICENSE` is the project's own; `THIRD_PARTY_NOTICES.md` reproduces the upstream MIT notice and lists which modules derive
  from `supermimic_vibed`. The long tables of the README are in `docs/reference.md`.
