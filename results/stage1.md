# Stage 1: contact tracker on reference targets

Run `acp_stage1_ft` (30,000 iterations on the published clips, then a fine-tune on the refined clips with `--target_min_steps 4`),
checkpoint 4000. Evaluation: every refined clip once from its first frame, first terrain variant, no noise (`python -m proacp.evaluation.clips`).

| run | seeds | iterations | time (h) | return ↑ | failed (of 296) ↓ | touch precision / recall, feet ↑ | placement err feet (cm) ↓ |
|---|---|---|---|---|---|---|---|
| `acp_stage1_ft`, ckpt 4000 | 1 | 30,000 on the published clips + 4,000 on the refined clips | 0.73 (fine-tune only) | 0.835 | 4 | 0.93 / 0.98, 0.91 / 0.98 | 10.0, 10.2 |

Later checkpoints of both runs are worse on this evaluation (8,000: 21 failed) although the training statistics keep improving; the
cause is not identified yet.

| folder | return | fail rate |
|---|---|---|
| motions-low-step | 0.867 | 0.018 |
| motions-high-step | 0.819 | 0.000 |
| motions-low-climb-76 | 0.828 | 0.000 |
| motions-high-climb-76 | 0.749 | 0.075 |
| motions-locomotion | 0.863 | 0.000 |
