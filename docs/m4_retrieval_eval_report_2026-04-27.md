# M4 Retrieval-Check Eval Report

**Date:** 2026-04-27
**Gate:** mean cohort outcome entropy ≥ 0.7 bits at k=64 on both holdouts.
**Result:** **FAIL**

## game_cold (n_queries=65705)

| Source | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|---|
| model | 0.651 | 0.678 | 0.692 | 0.700 | 0.704 |

### Per-minute entropy at headline k (model only)

| Minute | Entropy (bits) |
|---:|---:|
| 10 | 0.830 |
| 11 | 0.813 |
| 12 | 0.785 |
| 13 | 0.759 |
| 14 | 0.737 |
| 15 | 0.686 |
| 16 | 0.712 |
| 17 | 0.704 |
| 18 | 0.682 |
| 19 | 0.652 |
| 20 | 0.650 |
| 21 | 0.635 |
| 22 | 0.608 |
| 23 | 0.585 |
| 24 | 0.559 |
| 25 | 0.556 |

## player_cold (n_queries=73295)

| Source | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|---|
| model | 0.655 | 0.681 | 0.694 | 0.701 | 0.706 |

### Per-minute entropy at headline k (model only)

| Minute | Entropy (bits) |
|---:|---:|
| 10 | 0.836 |
| 11 | 0.815 |
| 12 | 0.788 |
| 13 | 0.765 |
| 14 | 0.743 |
| 15 | 0.686 |
| 16 | 0.715 |
| 17 | 0.709 |
| 18 | 0.688 |
| 19 | 0.652 |
| 20 | 0.653 |
| 21 | 0.635 |
| 22 | 0.603 |
| 23 | 0.587 |
| 24 | 0.552 |
| 25 | 0.551 |
