# howtowin.lol — Thesis Experiment Documentation

**Date:** 2026-04-23  
**Purpose:** Thesis-ready root experiment brief for the current howtowin.lol research stack.

## Experiment metadata

- **Project:** howtowin.lol
- **Experiment slug:** `howl/exp/canonical-world-model-v1`
- **Experiment type:** supervised root experiment
- **Owner:** Nyx
- **Status:** proposed
- **Priority:** high

## Objective

Build and validate a League of Legends world model for post-game teaching.

This experiment should answer one decision:

> Can the current world-model stack produce retrieval cohorts and lesson candidates that generalize to unseen players, remain leak-clean, and outperform or materially close the gap to the hand-crafted frame-feature retrieval baseline?

## Product context

howtowin.lol is not a generic stats dashboard. The target product is a post-game teaching system that should surface:

- **mistake lessons** — “from this state, your branch was usually losing”;
- **strength lessons** — “from this state, your branch was unusually winning”.

The experiment should optimize for teachable state structure, not only train loss.

## Current system snapshot

The current repo already contains:

- a three-stream model stack:
  - static game context,
  - player priors,
  - dynamic sequence;
- hybrid-time inputs:
  - anchor frames,
  - event tokens;
- an RSSM-style latent world model;
- downstream retrieval and lesson-candidate generation.

### Current implementation references

- dataset and labels: `code/model/dataset.py`
- tokenization and typed event payloads: `code/model/tokenizer.py`, `code/model/tokens.py`
- static, player, and dynamic encoders: `code/model/encoders.py`
- world model: `code/model/plan_b_model.py`, `code/model/rssm.py`
- training loop: `code/model/plan_b_train.py`
- outcome eval: `code/model/plan_b_eval.py`
- retrieval: `code/model/retrieval.py`, `code/model/m4_eval.py`
- lesson candidate generation: `code/model/lesson.py`

## Dataset

- **Source:** Riot match data + timelines parsed into local SQLite stores
- **Corpus size:** 12,846 matches
- **Players:** 21,007
- **Frame rows:** 3,654,210
- **Event rows:** 128,440
- **Raw match blobs:** 12,846

### Splits

- **game-cold holdout:** 64 matches
- **player-cold holdout:** 37 matches
- **overlap between holdouts:** 0 matches

## Inputs under study

### Static stream

- champion picks
- side assignment
- queue / region / time bucket
- numeric patch-conditioning vector

### Player stream

- crafted player-history features
- learnable per-player residual embedding

### Dynamic stream

- per-anchor participant frame features
- per-anchor macro / objective features
- ordered event-token windows
- inferred decision tokens such as recall / engage / disengage

## Current baseline to beat or preserve

Use the current documented Plan B + retrieval stack as the control.

### Outcome baseline

- **game-cold AUC@15:** 0.884
- **player-cold AUC@15:** 0.841
- **leak probe AUC@15:** 0.531

### Retrieval baseline

- **model entropy@64**
  - game-cold: 0.720
  - player-cold: 0.706
- **frame-features baseline entropy@64**
  - game-cold: 0.826
  - player-cold: 0.834

### Interpretation

- outcome signal exists;
- leak discipline is acceptable;
- retrieval is still the main bottleneck between the world model and the teaching surface.

## Hypothesis

If we improve state fidelity and preserve the current retrieval contract, then:

1. player-cold retrieval quality should improve materially;
2. outcome performance should stay at or near the current baseline;
3. lesson candidates should become more specific and more teachable.

## Primary metric

**Primary metric:** `player_cold_m4_entropy_at_64`

Rationale:

- retrieval quality is the main bottleneck between latent structure and teaching utility;
- player-cold is the stricter generalization test.

## Secondary metrics

- `game_cold_m4_entropy_at_64`
- `player_cold_auc_at_15`
- `game_cold_auc_at_15`
- `leak_probe_auc_at_15`
- rollout top-5 by step
- lesson candidate quality on a manual QA set

## Success criteria

This root experiment is considered a success if all of the following hold.

### Required

- `player_cold_m4_entropy_at_64 >= 0.75`
- `game_cold_m4_entropy_at_64 >= 0.75`
- leak probe remains `<= 0.55`
- player-cold AUC@15 does not regress by more than `0.02`
- game-cold AUC@15 does not regress by more than `0.02`

### Strong success

- retrieval gets within `0.05` bits of the frame-features baseline on both holdouts;
- lesson candidates become more decision-specific in manual review.

## Failure / stop conditions

Stop or mark failed if any of the following occur:

- leak probe rises above `0.55`;
- retrieval entropy falls below the current baseline by more than `0.03` bits;
- outcome AUC collapses on either holdout;
- training succeeds numerically but lesson candidates become more generic;
- an experiment changes split definitions or introduces holdout contamination;
- conclusions are based only on train loss.

## Experiment tree

### Root

`howl/exp/canonical-world-model-v1`

### Child 1 — baseline reproduction

**Goal**

- reproduce the current documented Plan B + retrieval numbers.

**Outputs**

- fresh outcome eval report,
- fresh retrieval eval report,
- exact command log,
- checkpoint hash.

### Child 2 — anchor / state-fidelity ablation

**Goal**

- test richer state features.

**Candidates**

- inventory summaries,
- warding summaries,
- health / power summaries,
- richer objective state.

### Child 3 — decision-semantics ablation

**Goal**

- improve inferred or factorized decision / event quality.

**Candidates**

- recall heuristic cleanup,
- engage / disengage heuristic refinement,
- payload-head weighting changes.

### Child 4 — retrieval-focused training ablation

**Goal**

- improve latent structure for retrieval.

**Candidates**

- loss-weight changes,
- representation regularization,
- curriculum or head-balancing changes.

### Child 5 — lesson QA

**Goal**

- test whether better retrieval actually yields better teaching examples.

**Outputs**

- 20-game manual review set,
- mistake + strength lesson assessment,
- failure taxonomy.

## Artifacts required from every run

- checkpoint path
- code commit SHA
- split file hashes
- config / hyperparameters
- training log
- outcome eval output
- retrieval eval output
- lesson samples from at least 3 held-out games
- short decision memo:
  - what changed,
  - what improved,
  - what regressed,
  - what to try next.

## Commands

Run from `code/`.

### Train

```bash
uv run python -m model.cli plan-b-train --epochs 30
```

### Outcome eval

```bash
uv run python -m model.cli plan-b-eval
```

### Build retrieval indexes

```bash
uv run python -m model.cli retrieval-build --baselines
```

### Retrieval eval

```bash
uv run python -m model.cli retrieval-eval --baselines
```

### Lesson candidate generation

```bash
uv run python -m model.cli lesson --match-id <MATCH_ID> --team blue
```

## Agent operating rules

- Do **not** change holdout splits.
- Do **not** use held-out matches in player aggregates.
- Do **not** optimize solely for train loss.
- Do **not** touch `secrets/`.
- Prefer small, interpretable ablations over wide random search at the root level.
- Every child run must state:
  - parent experiment,
  - one concrete change,
  - one expected effect,
  - one primary metric.
- If a run fails, produce a short failure diagnosis before launching the next run.

## Human review gate

Auto-generated experiment conclusions are suggestions, not final truth.

A human should approve:

- any change to canonical model direction;
- any public claim of improvement;
- any lesson-quality claim;
- any decision to replace the current baseline.

## Data handling note

This project includes research data, derived player features, and local secrets / config files.

When running in hosted or agentic environments:

- exclude `secrets/`;
- avoid uploading unnecessary raw private material;
- keep experiment artifacts scoped to model / retrieval outputs and reproducible metadata.

## Desired final output from Thesis

At the end of this root experiment tree, Thesis should produce:

1. the best validated checkpoint;
2. a ranked summary of ablations;
3. a clear answer on whether retrieval quality improved enough to justify downstream lesson work;
4. the next 3 highest-value experiments.
