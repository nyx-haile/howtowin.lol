# howtowin.lol — Thesis Experiment Documentation

**Date:** 2026-04-23  
**Purpose:** Thesis-ready root experiment brief for the current howtowin.lol research stack.

## Experiment metadata

- **Project:** howtowin.lol
- **Draft name:** `canonical-world-model-v1-root`
- **Experiment slug:** `howl/exp/canonical-world-model-v1-root`
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
- retrieval is still the main bottleneck between the world model and the teaching surface;
- **higher entropy@64 is better** because it indicates less degenerate retrieval cohorts at `k=64`;
- the current model underperforms the frame-features baseline, so the goal is to raise model entropy toward `0.826` / `0.834`, not lower it.

## Hypothesis

If we improve state fidelity and preserve the current retrieval contract, then:

1. `player_cold_m4_entropy_at_64` should improve by at least `0.02` bits over the current model baseline (`0.706` → `>= 0.726`);
2. outcome performance should stay within `0.02` AUC of the current baselines;
3. lesson candidates should become more specific and more teachable in human review.

**Important:** item 3 is a human-review hypothesis, not a machine-checkable pass/fail gate. The agent should generate lesson samples and a failure taxonomy, but should **not** self-grade lesson quality as success or failure.

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
- `rollout_top5_by_step` — next-event-type top-5 accuracy at rollout steps `1..3` from `plan-b-eval`; “by step” indexes the imagination horizon after seeding on the final real anchor
- lesson candidate quality on a manual QA set (**human review only**)

## Success criteria

This root experiment is considered a success if all of the following machine-checkable conditions hold.

### Required

- `player_cold_m4_entropy_at_64 >= 0.75`
- `game_cold_m4_entropy_at_64 >= 0.75`
- leak probe remains `<= 0.55`
- player-cold AUC@15 does not regress by more than `0.02`
- game-cold AUC@15 does not regress by more than `0.02`

### Strong success

- retrieval gets within `0.05` bits of the frame-features baseline on both holdouts;
- at least one Child `2`–`4` run improves `player_cold_m4_entropy_at_64` by `>= 0.02` bits over the reproduced control;
- human QA finds lesson candidates more decision-specific than the reproduced control.

## Failure / stop conditions

Stop or mark failed if any of the following occur:

- leak probe rises above `0.55`;
- retrieval entropy falls below the current model baseline by more than `0.03` bits on either holdout;
- outcome AUC@15 regresses by more than `0.02` on either holdout;
- Child 1 fails the baseline-reproduction tolerances defined below;
- an experiment changes split definitions or introduces holdout contamination;
- conclusions are based only on train loss.

## Experiment tree

### Root

`howl/exp/canonical-world-model-v1-root`

**Root experiment action**

- Execute **Child 1** directly as the root run.
- Compare the reproduced control numbers against the documented baselines using the tolerances below.
- Only if Child 1 passes should the root propose or launch **Children 2–4** as follow-up ablations.
- Children `2`–`4` may run in parallel because they test different causal hypotheses.
- **Child 5** is gated on at least one of Children `2`–`4` improving `player_cold_m4_entropy_at_64` over the reproduced control.

### Child 1 — baseline reproduction

**Goal**

- reproduce the current documented Plan B + retrieval numbers.

**Success criteria**

Reproduced numbers must fall within the following tolerances of the documented baselines:

- AUC metrics: within `±0.005`
- entropy metrics: within `±0.01`
- leak probe: must remain `<= 0.55`

Concretely, the reproduced run should land within tolerance for:

- game-cold AUC@15 (`0.884`)
- player-cold AUC@15 (`0.841`)
- leak probe AUC@15 (`0.531`)
- model entropy@64 on game-cold (`0.720`)
- model entropy@64 on player-cold (`0.706`)
- frame-features baseline entropy@64 on game-cold (`0.826`)
- frame-features baseline entropy@64 on player-cold (`0.834`)

If reproduction fails outside these tolerances, stop before ablations and diagnose likely causes:

- data version mismatch
- split file hash mismatch
- dependency version drift
- hardware / environment drift
- random-seed sensitivity

**Outputs**

- fresh outcome eval report
- fresh retrieval eval report
- exact command log
- checkpoint hash
- lockfile hash

### Child 2 — anchor / state-fidelity ablation

**Goal**

- test richer state features.

**Default first ablation**

- add inventory summary features (`item-slot occupancy` + `gold-efficiency ratio`) to the per-anchor participant frame.

**Expected effect**

- improve state fidelity around item power spikes and decision points with minimal engineering overhead.

**Primary metric**

- `player_cold_m4_entropy_at_64`

**Other candidate follow-ups**

- warding summaries
- health / power summaries
- richer objective state

### Child 3 — decision-semantics ablation

**Goal**

- improve inferred or factorized decision / event quality.

**Default first ablation**

- clean up the recall heuristic by cross-referencing base-arrival events with recall-start inference to reduce false positives, especially around deaths near base.

**Expected effect**

- fewer noisy decision tokens and a cleaner latent space for retrieval.

**Primary metric**

- `player_cold_m4_entropy_at_64`

**Other candidate follow-ups**

- engage / disengage heuristic refinement
- payload-head weighting changes

### Child 4 — retrieval-focused training ablation

**Goal**

- improve latent structure for retrieval.

**Default first ablation**

- increase retrieval-head loss weight by `2x` relative to the outcome-head loss weight.

**Expected effect**

- test whether the latent space is currently under-optimized for retrieval structure.

**Primary metric**

- `player_cold_m4_entropy_at_64`

**Other candidate follow-ups**

- representation regularization
- curriculum or head-balancing changes

### Child 5 — lesson QA

**Goal**

- test whether better retrieval actually yields better teaching examples.

**Gate**

- run only after at least one of Children `2`–`4` improves `player_cold_m4_entropy_at_64` over the reproduced control.

**Outputs**

- 20-game manual review set
- mistake + strength lesson assessment
- failure taxonomy
- side-by-side comparison between the reproduced control and the best-improved ablation

## Artifacts required from every run

- checkpoint path
- checkpoint SHA256
- code commit SHA
- `uv.lock` path and SHA256
- split file paths and SHA256 for:
  - `data/splits/plan_a_holdout.txt`
  - `data/splits/plan_b_cold_holdout.txt`
- copied split files in the artifact bundle if the runner does not automatically preserve repo-versioned inputs
- config / hyperparameters
- exact commands
- training log
- outcome eval output
- retrieval eval output
- lesson samples from at least `3` held-out games
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

Use the first `3` match IDs from `data/splits/plan_a_holdout.txt` for the minimum required sample set:

- `NA1_5439588777`
- `NA1_5516680826`
- `NA1_5517996204`

```bash
uv run python -m model.cli lesson --match-id NA1_5439588777 --team blue
uv run python -m model.cli lesson --match-id NA1_5516680826 --team blue
uv run python -m model.cli lesson --match-id NA1_5517996204 --team blue
```

## Reproducibility rules

- Use the committed `uv.lock` for every run. If it is missing or intentionally regenerated, record the new lockfile hash as an artifact before training.
- The current in-repo CLI does **not** expose a `--seed` flag. For non-seed-sensitivity runs, keep the training seed behavior fixed to the current code path and record it explicitly in artifacts. If a runner adds an explicit seed parameter before launch, pin it to `42`.
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

## Suggested monitor cadence

If the runner supports a monitor cron, use:

```text
*/5 * * * *
```

This run is likely to take on the order of `1`–`4` hours depending on hardware, so a `5`-minute monitor cadence is appropriate.

## Human review gate

Auto-generated experiment conclusions are suggestions, not final truth.

A human should approve:

- any change to canonical model direction;
- any public claim of improvement;
- any lesson-quality claim;
- any decision to replace the current baseline.

The agent should generate lesson samples and manual-review evidence, but should **not** self-assess “more specific,” “more teachable,” or “more generic” as an automatic pass/fail result.

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
4. the next `3` highest-value experiments.
