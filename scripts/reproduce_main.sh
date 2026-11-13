#!/bin/bash
# The runs behind the tables of the README, one after the other so the GPU is never shared (about 21 h on one GPU):
# Stage 1 (4.6 h) and its fine-tune on the refined clips (0.7 h), the planner on the frozen tracker (0.7 h), Stage 2 co-training
# (14.7 h), then the two evaluations. Run tools/download_data.sh and scripts/prepare_data.sh first.
# The fine-tune runs 4000 iterations: its ckpt_004000.pt is the reported checkpoint (later checkpoints of the original run evaluated worse).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
T=scripts/train.sh
[ -f logs/acp_stage1/last.pt ]    || $T stage1 acp_stage1 --iterations 30000
[ -f logs/acp_stage1_ft/ckpt_004000.pt ] || $T stage1 acp_stage1_ft --init logs/acp_stage1/last.pt \
    --motion_dir data/php_parkour_refined --target_min_steps 4 --iterations 4000
[ -f logs/acp_planner_ft/planner_020000.pt ] || $T planner_offline acp_planner_ft --tracker logs/acp_stage1_ft/ckpt_004000.pt
[ -f logs/acp_stage2/last.pt ]    || $T stage2 acp_stage2 --init logs/acp_stage1_ft/ckpt_004000.pt \
    --planner_init logs/acp_planner_ft/planner_020000.pt --iterations 24000
[ -f logs/acp_stage2_tw/ckpt_040000.pt ] || $T stage2 acp_stage2_tw --resume logs/acp_stage2/last.pt \
    --term_window 2 --eta_rew_max 0.58 --iterations 40000
# Stage 1 table: the tracker on the reference contact targets
$PY -m proacp.evaluation.clips --ckpt logs/acp_stage1_ft/ckpt_004000.pt
# Stage 2 table: the tracker on the planner's targets, with the training action noise and the +-2 s window
$PY -m proacp.evaluation.clips --ckpt logs/acp_stage2_tw/ckpt_040000.pt --planner logs/acp_stage2_tw/ckpt_040000.pt --train_noise action
