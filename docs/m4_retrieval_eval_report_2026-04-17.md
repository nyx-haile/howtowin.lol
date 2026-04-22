# M4 Retrieval-Check Eval Report

**Date:** 2026-04-17
**Gate:** mean cohort outcome entropy ≥ 0.7 bits at k=64 on both holdouts.
**Result:** **PASS on the necessary entropy gate**

**Historical note.** This report clears the headline `0.7`-bit threshold on both holdouts, but that threshold is only a necessary condition. The same tables show two unresolved interpretation questions: the model trails the frame-features baseline at every reported `k`, and the per-minute entropy curve is front-loaded rather than peaking mid-window. Follow-up beads were filed for both questions.

## game_cold (n_queries=933)

| Source | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|---|
| model | 0.552 | 0.658 | 0.720 | 0.734 | 0.755 |
| static_only | 0.000 | 0.000 | 0.008 | 0.653 | 0.889 |
| frame_features | 0.767 | 0.806 | 0.826 | 0.844 | 0.858 |
| random | 0.950 | 0.973 | 0.985 | 0.991 | 0.994 |

### Per-minute entropy at headline k (model only)

| Minute | Entropy (bits) |
|---:|---:|
| 10 | 0.817 |
| 11 | 0.802 |
| 12 | 0.793 |
| 13 | 0.771 |
| 14 | 0.758 |
| 15 | 0.690 |
| 16 | 0.765 |
| 17 | 0.712 |
| 18 | 0.712 |
| 19 | 0.709 |
| 20 | 0.714 |
| 21 | 0.711 |
| 22 | 0.658 |
| 23 | 0.596 |
| 24 | 0.613 |
| 25 | 0.623 |

## player_cold (n_queries=547)

| Source | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|---|
| model | 0.670 | 0.695 | 0.706 | 0.713 | 0.717 |
| static_only | 0.118 | 0.076 | 0.166 | 0.657 | 0.871 |
| frame_features | 0.768 | 0.807 | 0.834 | 0.853 | 0.869 |
| random | 0.951 | 0.973 | 0.986 | 0.991 | 0.994 |

### Per-minute entropy at headline k (model only)

| Minute | Entropy (bits) |
|---:|---:|
| 10 | 0.858 |
| 11 | 0.845 |
| 12 | 0.813 |
| 13 | 0.773 |
| 14 | 0.758 |
| 15 | 0.661 |
| 16 | 0.735 |
| 17 | 0.701 |
| 18 | 0.689 |
| 19 | 0.664 |
| 20 | 0.649 |
| 21 | 0.647 |
| 22 | 0.655 |
| 23 | 0.613 |
| 24 | 0.567 |
| 25 | 0.600 |
