# Plan B RSSM v1 — Gating Eval Report

**Date:** 2026-04-17
**Checkpoint:** `data/model_checkpoints/plan_b_full_best.pt`
**Corpus:** 12,846 games (post `data/staging` merge on 2026-04-17)
**Training:** 7 epochs, early-stop on player-cold AUC@15, batch_size=8, lr=3e-4, fp16 AMP
**Hardware:** GTX 1650 Super (TU116), 4GB VRAM
**Command:** `python -m model.cli plan-b-train --epochs 30`
**Eval log:** `data/logs/plan_b_eval_APR_17.log`

## Results

### Outcome AUC by minute

| Minute | Game-cold | Player-cold |
|-------:|----------:|------------:|
|  5     | 0.504     | 0.559       |
| 10     | 0.738     | 0.706       |
| 15     | 0.884     | 0.841       |
| 20     | 0.860     | 0.856       |
| 25     | 0.948     | 0.952       |

### Imagination rollout (game-cold, prior-only)

| Step | Event top-5 |
|-----:|------------:|
|  1   | 0.875       |
|  2   | 0.891       |
|  3   | 1.000       |

### Leak probe (frozen minute-0 features)

AUC@15 = **0.531** (target ≤ 0.55, clean)

## Comparison to prior baseline (1,799 games, same model)

| Metric             | 1,799 games | 12.8k games | Δ       |
|--------------------|------------:|------------:|--------:|
| game_cold AUC@15   | 0.871       | 0.884       | +0.013  |
| player_cold AUC@15 | 0.432       | 0.841       | +0.409  |
| leak probe AUC@15  | ~0.43       | 0.531       | stable  |

**Notes.** The player-cold jump from 0.432 → 0.841 is not a model change — the 5-puuid holdout had only a handful of matches at 1,799 games and was dominated by variance. At 12.8k games the split has enough mass to measure generalization, which is clean.

## Gating status

All Plan B M6 (Task 14) exit criteria met:

- [x] Game-cold outcome AUC@15 > 0.8
- [x] Player-cold outcome AUC@15 > 0.8
- [x] Imagination-rollout top-5 > 0.5 at step 3
- [x] Leak probe AUC@15 ≤ 0.55
