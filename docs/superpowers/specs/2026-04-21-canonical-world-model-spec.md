# Canonical Superpowers World-Model Spec

**Status:** Canonical source of truth for the model stack. Older specs and plans remain preserved as historical design records.
**Date:** 2026-04-21
**Preserves:** `2026-04-15-sequence-model-design.md`
**Reconciles:** `2026-04-15-plan-b-rssm-v1-design.md`, `2026-04-17-m4-retrieval-check-design.md`, the shipped Plan B / M4 eval reports, and the durable modeling ideas in `notes/`.

When this file disagrees with older downstream docs, **this file wins**. The older docs are not overwritten; they remain useful for milestone history and implementation context.

## Purpose

howtowin.lol is building a **League of Legends world model for post-game teaching**: "in your exact situation, the branch that won usually looked like X, and the branch that lost looked like Y." The end product is not a dashboard and not a rank-comparison engine. It is a lesson system built on top of a latent representation of game state, decisions, and eventual outcome.

The strongest requirement from the original design still stands: **positive and negative lessons are co-equal**. The system must support both:

- **mistake lessons** — the player took the losing branch in a split state;
- **strength lessons** — the player took the winning branch in a split state.

The model therefore has to represent not just what happened, but when a state was still meaningfully undecided and which decisions separated winners from losers.

## How to read this document

This spec uses three different kinds of statements on purpose:

1. **Canonical target architecture** — what the model is ultimately supposed to be.
2. **Allowed temporary proxies** — shortcuts that may be used for bounded milestones, but do **not** count as closure on the upstream question.
3. **Open questions** — things the stricter design still leaves undecided and that must be answered deliberately, not by implementation drift.

## Non-goals

Out of scope for this spec:

- percentile/rank-comparison products;
- pure correlation surfaces untethered from concrete in-game state;
- policy learning or reward optimization;
- UI, lesson copy generation, or spaced repetition mechanics;
- production retrieval serving details such as FAISS deployment.

## Design principles

1. **State fidelity before extra architectural cleverness.** The next big gains come from making the model see more of the game that is already stored, not from adding a fancier backbone to a thin state bundle.
2. **Three-stream conditioning is mandatory.** Static game context, player priors, and dynamic timeline state are all first-class inputs.
3. **Hybrid time is mandatory.** Frame anchors and event tokens are both needed; per-minute-only and per-event-only views are both too lossy.
4. **Decision/outcome separation is mandatory.** The model must distinguish what players chose from what the world produced next.
5. **Raw-first, patch-aware data discipline.** Raw match/timeline data is the durable asset; derived features must be rebuildable as patches change.
6. **Leak discipline is part of the architecture.** A model that only works through identity or future-information leakage is not a partial success.
7. **Temporary proxies must be named.** If a milestone uses a lighter target, thinner observation bundle, or pooled dynamics, the report must say so explicitly.

## Definitions

- **Anchor** — a fixed-timestep observation frame, currently minute cadence from Riot timeline frames.
- **Event token** — a meaningful time-stamped event between anchors, kept at true ordering and true timestamp.
- **Decision** — a player-controlled choice (purchase, level-up, ward, recall, engage/disengage, etc.).
- **Outcome** — a world consequence downstream of prior state and decisions (kill, tower, dragon, Baron, win/loss, etc.).
- **Proxy** — a temporary simplification accepted for a milestone only.
- **Fair test of the architecture** — an experiment whose inputs, targets, and evals are rich enough that success or failure can be attributed to the intended model design rather than to an impoverished approximation.

## Canonical target architecture

### 1. Inputs — three streams

#### Stream 1 — Static game context

This stream is **per game** and fixed at game start. It must include, at minimum:

- the 10 champion picks;
- blue/red side assignment per participant;
- queue and region;
- time-of-day or equivalent session-context scalar/categorical features;
- rich numeric patch conditioning derived from the actual patch parameters relevant to the game.

The patch-conditioning requirement is unchanged from the original spec: it exists so the model can eventually support counterfactual patch interventions, not just memorize patch IDs.

Canonical conditioning rule:

- static context is encoded once per game;
- the encoded static context is available to the sequence model at **every anchor/event step** via cross-attention;
- cross-attention is required, not merely preferred: a seed-only conditioner (e.g. `static_to_h`) does not satisfy per-step access and must not be reported as equivalent.

**Important:** patch-version scaffolding by itself is **not** full Stream-1 coverage and must never be reported as such.

#### Stream 2 — Player models

This stream is **per participant** and represents identity/skill priors.

Required structure:

- hand-crafted player/champion/role history features;
- a learnable per-puuid residual embedding, zero-safe for unseen players;
- fusion of the player representation into that participant's dynamic state at every step.

Required leak discipline:

- any rolling or aggregate player feature must exclude held-out matches;
- for held-out matches, it must also exclude future matches from that player's perspective.

#### Stream 3 — Dynamic sequence

This stream is the core timeline and has two token types.

##### 3A. Anchor observations

Anchors are the model's observation points and must be rich enough to describe the actual game state. The notes and the stored raw timeline data both point toward an **axis-wise time × participant × stat** representation with team/objective summaries, so the canonical anchor bundle is broader than the current thin `10 × 6` feature proxy.

At minimum, the canonical anchor observation should cover:

- **per participant economy/resources:** total gold, current gold, xp, level, CS, jungle CS;
- **per participant combat scoreboard:** kills, deaths, assists;
- **per participant spatial state:** position x/y and a stable representation of movement/location context;
- **per participant combat/resource stats when available from frames:** health/power and key damage/champion-stat summaries;
- **inventory / warding state summaries** when derivable without leak;
- **team/objective macro state:** gold diff, xp diff, kill diff, tower/plate state, neutral objective state (drakes, grubs, Herald, Baron, Elder), and other high-signal macro summaries such as team centroids/spread if helpful.

The exact feature list is still an implementation question, but the requirement is not: the model must carry substantially richer anchor state than the current minimal proxy if we want a fair test of the notes and the original spec.

##### 3B. Event tokens

Event tokens remain required between anchors and must preserve **true time order**.

Canonical event semantics:

- one token per meaningful event;
- each token carries at least **event type, actor slot, target slot where applicable, and subtype/payload where applicable**;
- payload examples include item identity, skill slot, monster subtype, building lane/tower subtype, ward subtype, and other distinctions that matter for teaching or counterfactuals;
- inferred decision tokens such as `RECALL`, `ENGAGE`, and `DISENGAGE` remain part of the design.

A plain 11-way event-type proxy is acceptable only as a temporary milestone shortcut. It is not the end-state event objective.

### 2. Decision/outcome contract

The original two-tier idea remains correct and stays locked.

**Decisions** include, at minimum:

- item purchases;
- skill level-ups;
- ward placements / control-ward placements;
- recall;
- engage / disengage.

**Outcomes** include, at minimum:

- champion kills;
- building kills;
- elite monster kills;
- champion special kills;
- eventual game win/loss.

Canonical modeling goal:

- the latent state should support the question "from this state, conditioned on these decisions, what outcomes tend to follow?"
- retrieval and lesson generation then look for states whose nearby cohorts split by decisions into different outcomes.

## Sequence model contract

### Why RSSM remains the target family

The reason for keeping an RSSM-style model is unchanged: future balance-simulation and counterfactual rollout work need a model that can be advanced in latent space under interventions. A plain teacher-forced transformer may be acceptable as a baseline, but it does not settle the imagination-rollout requirement.

### Canonical recurrence

The canonical model has:

- deterministic recurrent state `h`;
- stochastic latent `z`;
- prior `p(z | h)`;
- posterior `q(z | h, o)` at anchor observation points;
- KL regularization with anti-collapse measures such as free bits / balanced KL.

Canonical update rule:

- event tokens update the recurrent state **between anchors**;
- anchor observations drive the posterior update;
- static context conditions the recurrent dynamics and/or posterior at every step;
- player priors stay fused into participant-local dynamic representations throughout the sequence.

A pooled per-anchor action summary may still be used as a temporary proof-of-life approximation, but it does **not** answer the upstream question of whether the world model is correctly using event-level dynamics.

### Representation consumed by heads and retrieval

The canonical predictive representation at anchor `t` is:

```text
r_t = [ h_t || z_t ]
```

Rules:

- decoder heads consume `r_t` rather than only `z_t`;
- retrieval uses `[ h_t || μ_q(z_t) ]` as its deterministic key;
- if an implementation deviates from this, that deviation must be called out explicitly.

### Decoder heads

All major heads remain required.

1. **Next-event head**
   - Semantic target: the next meaningful event identity, including actor/target and important subtype/payload information.
   - Implementation may factorize this target into multiple conditionals instead of one huge flat categorical.
   - A coarse event-type-only proxy is allowed only for bounded milestones.

2. **Next-observation / next-frame head**
   - Predicts the next anchor's observable state, including participant deltas and relevant macro/objective deltas.
   - Grounds the latent in concrete game state.

3. **Outcome head**
   - Predicts eventual game outcome from each anchor.
   - This is a primary causal objective because retrieval and teaching depend on outcome-sensitive latent structure.

4. **Next-decision head**
   - Predicts per-participant next decisions.
   - Necessary for later decision-diff explanations and counterfactual lesson work.

5. **Optional auxiliary heads**
   - Team macro summaries, objective-state transitions, positioning summaries, or other targets derived from `features.py` / notes may be added.
   - They are useful only if they sharpen state representation; they do not replace the four core heads above.

### Training priorities

This spec intentionally locks the **ordering of priorities**, not exact numeric coefficients:

- next-event and outcome are the primary objectives;
- next-decision and next-observation are secondary grounding objectives;
- KL regularization must be strong enough to keep the latent meaningful and weak enough to avoid collapse in the current data regime.

Implementation plans must report the literal code weights used and must not claim equivalence to spec-level language unless demonstrated.

### Milestone pass/fail thresholds

Concrete pass/fail thresholds (e.g., outcome AUC, entropy gate, top-5 accuracy) are **not** defined in this spec. They live in per-milestone design documents and must be set against a model trained on the canonical inputs — thin-proxy thresholds from earlier system states are not valid baselines for architecture-validation claims.

### Rollout contract

If the model is evaluated or trained for imagination rollouts, the rollout path must obey the same state semantics as the teacher-forced path.

Required rules:

- prior rollouts must seed from the true saved recurrent state for the chosen anchor, not from a reset approximation;
- imagined step `t+n` must be supervised, when labels are available, against the **true future step `t+n` labels**, not a repeated last-observed label;
- rollout reporting must make clear whether the result is a true latent rollout or a temporary approximation.

## Retrieval and teaching contracts

### Retrieval (M4)

Retrieval remains a downstream consumer of the learned latent, not a separate modeling philosophy.

Canonical retrieval key:

```text
key(g, t) = [ h_t || μ_q(z_t) ]
```

Canonical retrieval eval requirements:

- query held-out mid-game anchors (minutes 10–25);
- index only training-split anchors;
- whiten keys using training-corpus statistics;
- report both game-cold and player-cold results.

Interpretation rules:

- **necessary gate:** mean cohort outcome entropy at headline `k` must be at least `0.7` bits on both holdouts;
- **not sufficient by itself:** the result must also be interpreted against random-k and strong hand-crafted baselines such as frame features;
- **too close to random** means the cohort is loose and unhelpful;
- **too one-sided** means the cohort is unteachable;
- a model that clears `0.7` but loses clearly to a simple frame-feature baseline is a provisional pass, not closure on representation quality.

### Teaching surface (M5)

The first teaching surface is only real when it can do all three of these at once:

1. find a high-signal anchor;
2. identify whether the held-out game is on the winning or losing side of the split;
3. explain what decisions separated the two branches.

Anchor selection alone is not enough.

Strength lessons remain a first-class requirement, not optional polish.

## Allowed temporary proxies and their exit conditions

| Area | Canonical target | Allowed temporary proxy | Not enough to claim | Retrain needed when removed? |
|---|---|---|---|---|
| Static stream | Full game-start context conditioned every step | Patch scaffolding or otherwise thin static vector | Fair test of the three-stream design | Yes |
| Anchor observations | Rich participant + team/objective state | Thin per-anchor frame subset | Fair test of state fidelity | Yes |
| Event semantics | Actor/target/subtype-aware event identity | 11-way event-type window proxy | Teaching-ready world model | Yes |
| Between-anchor dynamics | Event-token recurrence | Pooled per-anchor action summary | Trustworthy world-model / rollout claim | Yes |
| Head input | `r_t = [h_t || z_t]` | Heads on `z_t` only | Retrieval and decoder-contract alignment | Yes |
| Rollout supervision | True latent rollout with true future labels | Approximate rollout using reset state or repeated labels | Counterfactual-readiness claim | Yes |
| Retrieval eval | Entropy + baseline contrast + per-minute interpretation | Entropy threshold alone | M5 readiness | No, if only reporting changes |
| Teaching surface | Anchor + polarity + decision-diff explanation | Anchor ranking only | User-facing lesson quality | Sometimes; depends on whether missing signal is model or analysis only |

## Evaluation contract

### Leak and generalization checks

Every serious training run must report:

- game-cold holdout metrics;
- player-cold holdout metrics;
- a frozen-minute-0 leak probe or equivalent static/player-only sanity check.

### Proof-of-life vs architecture-validation

The downstream docs were right to separate these two layers. This spec keeps that distinction explicit.

#### A. Proof-of-life checks

These answer only: "is there real signal here?"

- outcome AUC materially above chance by mid-game on both holdouts;
- player-cold performance that is not collapsing relative to game-cold;
- clean leak probe;
- rollout accuracy above trivial chance if rollout is being exercised at all.

Passing these checks means the project should continue. It does **not** mean the canonical architecture question is settled.

#### B. Architecture-validation checks

These answer: "is this a fair test of the intended world model?"

Examples of the stricter checks that should be satisfied before claiming closure on the architecture:

- outcome AUC should generally strengthen with game time rather than showing pathological flatness;
- rollout quality should not improve with forecast horizon for accidental reasons;
- retrieval should beat random decisively and be competitive with or better than strong hand-crafted baselines;
- reported results should name any still-active proxies in inputs, targets, or dynamics.

## Retraining implications

The answer to "do we need to rerun training?" depends on what changed.

**Retraining is required** when changing any of the following:

- static-stream feature content or conditioning path;
- anchor observation schema;
- event semantics / target factorization;
- recurrence granularity between anchors;
- head input contract (`z_t` only vs `[h_t || z_t]`);
- rollout training target or seeding behavior;
- major loss composition.

**Retraining is not automatically required** for:

- doc-only clarifications;
- offline eval/reporting cleanups on a fixed checkpoint;
- additional retrieval baselines computed from existing saved representations;
- lesson-surface analysis changes that do not alter the model or embeddings.

## Open questions that still need deliberate answers

### Must be answered before the next architecture-alignment retrain

1. **Exact anchor observation schema.** Which participant, team, and objective features are in the first full-fidelity bundle?
2. **Exact event factorization.** Flat categorical, structured factorization, or mixed coarse/fine heads?
3. **Static conditioning mechanism.** Cross-attention, FiLM/gating, or another equivalent per-step conditioner?
4. **Recurrence staging plan.** Do we jump directly to event-token recurrence, or stage through a richer pooled approximation one last time?
5. **Head/input alignment.** If retrieval uses `[h || μ(z)]`, are all decoder heads moved to the same contract in the same retrain?

### Can remain plan-level for now

- exact hidden sizes and latent dimension;
- continuous vs categorical latent choice;
- exact KL coefficient schedule;
- exact auxiliary-head set;
- exact retrieval headline `k`, as long as a sweep is reported;
- exact optimizer schedule and early-stopping details.

## Not revisiting

Locked unless new evidence appears:

- positive and negative lessons are co-equal product goals;
- three-stream input shape is the correct high-level design;
- hybrid time (anchors + event tokens) is required;
- decision/outcome separation is required;
- RSSM-style latent dynamics remain the target family for counterfactual readiness;
- patch conditioning must stay numeric-rich in the end-state, not collapse to categorical patch ID only;
- the model should be improved first by increasing state fidelity, not by pretending the current thin proxies answer the upstream design.
