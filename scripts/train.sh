#!/bin/bash
# Run one training command with its log folder, keeping stdout next to the checkpoints.
#   scripts/train.sh <command> <run_name> [extra args...]
#   scripts/train.sh stage1 acp_stage1 --iterations 30000
# <command> is a module of proacp.train: stage1, planner_offline, stage2 or stage3. The run writes
# logs/<run_name>/ (checkpoints, args.json, train_log.csv, stdout.log); `--resume logs/<run_name>/last.pt` continues it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
CMD=${1:?command (stage1 | planner_offline | stage2 | stage3)}; RUN=${2:?run name}; shift 2
case "$CMD" in stage1|planner_offline|stage2|stage3) ;; *) echo "unknown command $CMD"; exit 1 ;; esac
mkdir -p "logs/$RUN"
$PY -m "proacp.train.$CMD" --log_dir "logs/$RUN" "$@" 2>&1 | tee -a "logs/$RUN/stdout.log"
