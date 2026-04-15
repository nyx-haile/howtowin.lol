# Plan B — RSSM v1 Design

**Status:** Design approved. Ready for implementation plan.
**Date:** 2026-04-15
**Predecessor:** `2026-04-15-sequence-model-design.md` (architecture baseline); `2026-04-15-plan-a-data-pipeline-and-baseline-classifier.md` (data pipeline + baseline, shipped).

## Purpose

Plan B implements **Milestone 3 (RSSM v1)** from the approved core sequence model design, amended to reflect lessons from Plan A — in particular the static-vector label leak that invalidated Plan A's held-out top-5 and motivated stricter leak discipline across all input streams.

Framing: **proof of life in a data-starved regime.** The corpus is 731 games (Plan A) growing toward a 3000-game target during Plan B. That is Dreamer-tiny. Plan B does not need state-of-the-art numbers; it needs to demonstrably carry causal signal that survives a leak probe and a player-cold-start evaluation. Passing unlocks corpus scaling and Plan C (retrieval). Failing stops the project before we waste compute on more data.

## Scope

**In:**
- Full RSSM (deterministic GRU + stochastic latent + prior + posterior) as specified in the core sequence model design.
- All four decoder heads (next-event, next-frame, outcome, next-decision) with revised loss weights.
- Prior-network training that supports multi-step imagination rollouts.
- Stream-2 leak audit and player-cold-start secondary holdout.
- Eval harness covering by-minute outcome AUC, imagination-rollout accuracy, and game-cold vs player-cold comparison.

**Out (deferred):**
- Retrieval over `z_t` — Plan C.
- Teaching surface — Plan D.
- Balance-simulation rollouts — Milestone 6. Static-vector scaffolding is prepared but not exercised.
- Bot generation via next-decision imagination rollout — head is trained and evaluated as an observability metric; no user-facing bot in Plan B.
- Cross-patch generalization — corpus is one patch window; expanded patch range waits for corpus growth.
- Policy / reward learning — still out.

## Amendments from the core sequence model design

The core design from `2026-04-15-sequence-model-design.md` stands except where amended below. Everything unlisted (three-stream input, two-tier decision/outcome split, hybrid-time dynamic sequence, RSSM over plain transformer, per-participant state fusion, cross-attended static context) is unchanged.

### A1. Loss-head weight rebalance

The core spec put 95% of loss weight on next-event. Plan A's leak demonstrated that next-event is vulnerable to pipeline leak and is a relatively easy metric to game. Retrieval (Plan C) consumes `z_t`'s ability to predict outcome, not its ability to predict next event. Step-by-step simulation (confirmed goal beyond Plan B) requires next-event *and* next-decision to be strong.

Revised weights:

| Head | Weight | Rationale |
|---|---|---|
| Outcome (win/loss) | 0.35 | Causal anchor; what retrieval consumes; hardest to game. |
| Next-event | 0.35 | Needed for simulation and bot generation (future). |
| Next-decision (per participant) | 0.15 | Bot-head readiness; ties decision stream into world model. |
| Next-frame features | 0.10 | Grounding `z_t` in observable state. |
| KL regularizer on `z_t` | 0.05 | Prevents trivial posterior; see §A4 for data-starved tuning. |

### A2. Prior-network rollout reconstruction

Dreamer-standard training leaves the prior `p(z_t|h_t)` supervised only via KL to the posterior. For imagination rollouts to work beyond single steps, the prior must carry event-predictive information on its own.

Add a **prior-rollout reconstruction loss**: at each anchor during training, sample a 3-step imagination rollout from the prior (no posterior updates for those steps), apply the next-event and outcome decoders on the rolled-out latents, and include their loss with weight 0.05–0.10 blended across the three forecast steps. Weight is small so teacher-forced training dominates, but non-zero so the prior learns to stand alone.

Concretely:
- Rollout step 1: weight 0.05.
- Rollout step 2: weight 0.03.
- Rollout step 3: weight 0.02.

Total rollout-loss budget ≈ 0.10, balanced against the 0.35 next-event + 0.35 outcome teacher-forced loss.

### A3. Static conditioning vector — scaffolding mode

Plan A's static vector leaked end-of-game item builds (commit `81dfdfa` fixed it). The replacement — a patch-level aggregate — is near-informationless because it varies only with patch, and the explicit version triple already encodes that.

Plan B keeps the static-vector infrastructure intact for Milestone 6 (balance simulation) but acknowledges it is currently scaffolding:

- **`PATCH_VECTOR_DIM` grows from 256 to 1024.** 256 was undersized for a real per-item-and-champ schema in Milestone 6 (Data Dragon has ~160 champs × ~10 stats plus ~200 items × ~20 stats). Retraining is already required from the leak fix, so shape change has zero migration cost.
- **`PATCH_VECTOR_MODE` config flag.** `"scaffolding"` zeros champion-stat and item-aggregate regions at vector construction, leaving only the version triple alive. `"full"` exercises the full schema (Milestone 6). Default during Plan B: `"scaffolding"`.
- **Rationale:** we do not ship a misleadingly "rich" static vector into a retrieval index. If a feature region is not actually informative right now, its slots are explicitly zero rather than subtly noisy.

### A4. KL and latent sizing for data-starved regime

With 731–3000 games, the stochastic latent is under-supported. Expect posterior collapse as the **default failure mode**, not an edge case.

- **KL weight:** start at 0.01 (Dreamer's typical ~1.0 is miscalibrated for this scale).
- **Free-bits:** 0.5 per latent dim.
- **Latent dimension `z_t`:** 32 continuous. Smaller than Dreamer-typical; larger wastes capacity on noise at current scale.
- **Monitoring:** KL per anchor logged every epoch. Near-zero = collapse (regularizer too weak *or* posterior not reading the observation). Very high = regularizer too strong. Rising slowly = healthy learning.

### A5. Training budget and early stopping

- **Max epochs:** 30.
- **Early stopping criterion:** player-cold-start outcome AUC @ minute 15 (not game-cold, not loss). Stop if no improvement for 5 epochs.
- **Why player-cold:** game-cold optimizes for identity memorization; player-cold reflects the actual target of generalization. Using player-cold for early stopping also removes the temptation to overtune to game-cold.

## Data & leak discipline

Before Plan B trains a single step, two gates must be passed:

### B1. Stream-2 leak audit

`player_feature_vector(puuid)` today reads the full corpus. Features like rolling winrate, KDA, CS@10 almost certainly include games that are in the held-out set — the player's future, from the model's perspective.

Required changes:
- Accept optional `exclude_match_ids: set[str]` parameter.
- When computing any rolling or aggregate feature, filter out `exclude_match_ids` *and* any games chronologically after those matches for that puuid.
- Dataset construction passes the union of (game-cold holdout, player-cold holdout) as the exclusion set.

Regression test (copy pattern from `test_item_slots_are_patch_aggregate_not_match_build`): a puuid's feature vector must differ when a recent game is added vs. removed from the excluded set. Ship the test before ship the fix.

### B2. Player-cold-start secondary holdout

Keep the existing deterministic SHA1(match_id) % 10 == 0 holdout as primary (64 games, game-cold generalization). Add a second deterministic split:

- **Selection:** pick ~5 puuids deterministically (hash `sorted(puuids)` by SHA1, take top-5 by hash). Reserve every match containing any of those 5 puuids for the player-cold holdout.
- **File:** `data/splits/plan_b_cold_holdout.txt`.
- **Non-overlap:** the cold-holdout match_ids are also excluded from the training split. Expected training corpus shrinks from ~667 → ~640 games.
- **Reporting:** every eval invocation reports both "game-cold" and "player-cold" metrics side-by-side.

### B3. Frozen-minute-0 leak probe

Sanity check that runs in every eval: feed only the static + player streams (zero out the entire dynamic sequence). Predict outcome. Target: ≈0.5 AUC. If the model beats chance here, something in static/player streams leaks endgame info.

## Target metrics for Milestone 3

Primary gating metrics — Plan B *must* hit these to be considered passing:

1. **Outcome AUC per minute — curve check.**
   - Per-minute AUC at minutes 5, 10, 15, 20, 25 on both game-cold and player-cold holdouts.
   - Monotonically non-decreasing with game time on *both* holdouts.
   - Game-cold target: ≥0.55 at min 5, ≥0.70 at min 15, ≥0.85 at min 25.
   - Player-cold target: ≥80% of game-cold at each timepoint (i.e., if game-cold min-15 is 0.75, player-cold must be ≥0.60).
   - **Flat curve is an automatic fail** — model is not using mid-game state.

2. **Imagination-rollout event top-5.**
   - Teacher-force the RSSM up to minute `t`, then roll the prior forward 3 anchor steps with no posterior updates.
   - Report top-5 accuracy at rollout step 1, 2, and 3.
   - Target: step-1 ≥ step-2 ≥ step-3. Step-3 must beat uniform-11 chance (0.45); collapse to uniform is an architectural fail.

3. **Game-cold vs player-cold gap on outcome AUC@15.**
   - Player-cold / game-cold ≥ 0.80.
   - A bigger gap signals identity memorization dominates state understanding.

Secondary (observability, no pass/fail):
- Frame-feature MSE per minute.
- KL divergence per anchor, logged per epoch.
- Next-decision top-1 accuracy per participant.
- Top-5 of 11 event types — *floor only*; anything below 0.70 is broken.

Leak probe (automatic fail if triggered):
- Frozen-minute-0 outcome AUC ≤ 0.55 at minute 25.

## Architecture components (consolidated)

For implementation reference. Unchanged from the core spec except where amended above.

- **Static encoder.** MLP: `PATCH_VECTOR_DIM=1024` → 512 → 512. Output cross-attends into the RSSM at every anchor step. `PATCH_VECTOR_MODE` flag controls whether non-version slots are zeroed.
- **Player encoder.** MLP over `PLAYER_FEATURE_DIM=16` crafted features → 128 → `D_PLAYER`. Learnable per-puuid residual of `D_PLAYER` dims, zero-initialized, `padding_idx=0` for unseen players. Residual added to MLP output. Fused into each participant's state vector in the dynamic stream. `D_PLAYER` carries forward from Plan A (256) unless the implementation plan pins it lower; consistency with Plan A's existing encoder is the default.
- **Dynamic stream embedder.** Token embedding (vocab 19) + actor slot embedding (`NUM_SLOTS=11`) + sinusoidal positional encoding on `timestamp_ms`. Participant player-embedding gathered at positions with `actor > 0`.
- **RSSM core.**
  - `h_t`: GRU hidden, dimension 512.
  - `z_t`: stochastic latent, dimension 32, continuous (isotropic Gaussian).
  - Prior `p(z_t | h_t)`: MLP head on `h_t` producing (μ, log σ).
  - Posterior `q(z_t | h_t, o_t)`: MLP head on concat(`h_t`, anchor observation features) producing (μ, log σ).
  - KL-balanced loss: stop-gradient on posterior for KL-to-prior, stop-gradient on prior for KL-to-posterior; standard Dreamer pattern.
  - Free-bits: 0.5 per latent dim, applied per-sample per-step.
- **Decoder heads (all distributional, operate on `z_t`):**
  1. Next-event: multi-hot BCE over 11 event types for the next-minute window (Plan A's label shape, retained).
  2. Next-frame features: per-participant μ and σ over gold/XP/level/position deltas.
  3. Outcome: scalar logit for game win at each anchor (supervised with game-end label, broadcast).
  4. Next-decision: categorical per participant over decision types (ITEM_PURCHASED, SKILL_LEVEL_UP, WARD_PLACED, RECALL, ENGAGE, DISENGAGE).

## Tests (Plan B adds to Plan A's suite)

In addition to Plan A's 53 passing tests:

- Stream-2 leak regression (§B1).
- Player-cold split determinism: `build_player_cold_holdout()` returns identical match-id set for identical input corpus.
- Prior rollout shape test: calling `rollout_prior(n=3)` produces the right-shaped tensors.
- Leak-probe smoke test: `eval_frozen_m0` returns finite AUC on a tiny fixture.
- RSSM unit tests: posterior/prior output shapes, KL loss non-negativity, free-bits clamping.

## Implementation principles

Carried from Plan A:
- Exact file paths in the plan.
- TDD per task; test → implementation → commit.
- Frequent, small commits.
- No placeholders in the implementation plan (YAGNI, DRY, show real code, not "similar to above").

Plan-B-specific:
- Every new model component ships with a shape-check unit test before any training use.
- Every new feature that could leak ships with a regression test *in the same commit*.
- Training CLI flags default to scaffolding-safe configurations (e.g. `--patch-vector-mode scaffolding`).

## Open questions (resolved during implementation plan)

These are pinned during the plan-writing phase from the spec's perspective; if new evidence emerges during implementation, revisit.

- Exact structure of `ANCHOR_OBSERVATION_FEATURES` fed to the posterior.
- Whether to use categorical latent (Dreamer-V2/V3 style) instead of continuous 32-dim Gaussian. **Default: Gaussian.** Categorical adds implementation complexity that is not warranted at this corpus scale.
- Rollout-loss weighting schedule (fixed vs ramp). **Default: fixed weights 0.05/0.03/0.02.**
- Gradient-clipping norm. **Default: 1.0** (carried from Plan A).

## Not revisiting

Locked; do not reopen without new evidence:
- All core sequence model design "Not revisiting" items remain locked.
- Additionally locked by Plan B:
  - Loss head weights (§A1).
  - Scaffolding mode for static vector (§A3).
  - Player-cold holdout methodology (§B2).
  - Outcome AUC as primary gating metric (§metrics).
