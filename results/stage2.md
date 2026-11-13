# Stage 2: planner and co-training

## Planner, offline on the frozen tracker

`acp_planner_ft`, `planner_020000.pt` (20,000 steps, 0.7 h) on the Stage 1 tracker `ckpt_004000`. The validation set is every 10th clip
of each folder.

| body | F1 ↑ | median top-1 error (cm) ↓ | within 0.1 m ↑ |
|---|---|---|---|
| feet (L / R) | 0.90 / 0.89 | 7.2 / 7.2 | 70 % / 70 % |
| knees (L / R) | 0.95 / 0.73 | 7.9 / 7.7 | 69 % / 68 % |
| hands (L / R) | 0.86 / 0.92 | 8.2 / 7.2 | 64 % / 77 % |
| elbows (L / R) | — | — | — |
| pelvis | — | — | — |

Elbows and pelvis hardly touch the terrain in these clips: no validation contact.

## Co-training

`acp_stage2` / `acp_stage2_tw` (`train.stage2`, 40,000 iterations from the Stage 1 tracker): η_obs ramps over 2k–22k iterations, η_rew over
12k–32k. From 24k on, the run continues with `--term_window 2 --eta_rew_max 0.58`. The tracker follows the planner's targets only
(η_obs = 1), with the ±2 s termination window and the training action noise (σ = 0.1) (`--train_noise action`).

| checkpoint | return ↑ | failed (of 296) ↓ | high climb | low climb | high step | low step | locomotion |
|---|---|---|---|---|---|---|---|
| 24,000 | 0.59 | 10 | 12.5 % | 7.5 % | 0 % | 0 % | 2 % |
| 40,000 | 0.56 | 8 | 10 % | 7.5 % | 0 % | 0 % | 1 % |

Without the training action noise the same checkpoints fail far more: 165 of 296 at 40k, and the Stage 1 tracker at 44k fails 170 instead
of 7. Observation, target or scan noise alone does not recover them; σ = 0.05 is not enough (101 failed).
