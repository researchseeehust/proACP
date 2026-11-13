# Results

The tables of the [README](../README.md) as files, one per stage:

* [`stage1.md`](stage1.md): the contact tracker on reference targets (`acp_stage1_ft`, checkpoint 4000)
* [`stage2.md`](stage2.md): the planner trained offline, and the tracker driven by the planner's targets (`acp_stage2_tw`, checkpoint 40000)

All numbers are from single-seed runs of this code (`experiments/runs.txt`) on one RTX PRO 5000 Blackwell. **The per-run logs and
checkpoints are not part of the repository**, so these tables cannot be regenerated from files here; `scripts/reproduce_main.sh`
reruns the training and the two evaluations, and each evaluation writes a json next to its checkpoint
(`python -m proacp.evaluation.clips --ckpt <checkpoint>`).
