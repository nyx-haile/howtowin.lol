# Sequence Model Design — howtowin.lol Core

**Status:** Design approved. Ready for implementation plan.
**Date:** 2026-04-15

## Purpose

howtowin.lol builds a **League of Legends world model** for causal, per-game teaching advice — "in your exact situation, doing X would have won the game." The advice surface is post-game review styled like Duolingo: bite-sized lessons anchored on critical-point divergences between games that reached near-identical states.

**Positive psychology is a first-class goal, co-equal to mistake-correction.** The teaching surface highlights *strengths* as well as mistakes — "you made the right call in a situation where most players lost" is often more valuable than "you made the wrong call in a situation most players won." A player who only sees their errors stops playing; a player who sees their edges keeps learning. This symmetry propagates down into critical-point detection: the model must detect both losing-side-of-split and winning-side-of-split anchors.

This document specifies the **core sequence model** that powers that surface. It does *not* cover the retrieval layer, the lesson generator, or the UI — those are downstream consumers of the latent state `z_t` produced here.

## Non-goals

Explicitly **out of scope** for this design:

- Rank-relative percentile comparisons ("you're 28th percentile Diamond").
- Correlational "what Challengers do differently" advice untethered from the player's situation.
- Stats dashboards.
- Policy / reinforcement learning. No reward head. No action-selection objective.
- The retrieval layer, lesson generator, spaced-repetition engine — these consume `z_t` and are designed separately.

## Principles

Four commitments that constrain every downstream decision:

1. **Latent-space prediction.** Predict in a learned compressed state `z_t`, not raw observations.
2. **Distributional output.** Heads emit distributions, not point estimates.
3. **Multi-objective training.** Multiple decoder heads train `z_t` to carry everything the system needs.
4. **Action-conditioned.** Dynamics condition on player decisions, not just state.

## Inputs — three streams

The model consumes three heterogeneous input streams. Each is encoded separately; outputs are fused into the sequence model.

### Stream 1 — Static game context

Per-game, fixed at game start. Conditions every timestep via cross-attention.

- **Picks** — 10 × champion ID (embedded).
- **Sides** — blue/red per participant.
- **Queue, region, first-dragon type, time of day** — categorical/scalar features.
- **Patch as rich numeric conditioning** — a dense vector of the actual parameters that changed this patch: item stats (AD, AP, crit, cost) for items present in the game, champ stats (Q/W/E/R ratios, base stats at played levels) for the 10 picks, monster stats (drake HP, baron buff values), tower plate gold. Sourced from Riot Data Dragon per patch.
  - **Why dense numeric, not categorical patch ID:** the balance-simulation use case (Milestone 6) requires extrapolation to unseen patch values — "what if IE crit was 30%?" Categorical IDs can't extrapolate; numeric parameters can.
  - Projected through a small encoder before conditioning.

### Stream 2 — Player models (10, one per participant)

Per-participant identity/skill prior. Fused into each participant's state vector in the dynamic stream.

- **Hand-crafted features** → small MLP: rank, mastery points on this champ, games-on-champ, historical KDA by role, winrate last N games, average CS@10, etc.
- **Learnable residual per puuid**: a learnable delta vector, zero-initialized, zero for unseen players.
  - Captures playstyle nuance the crafted features miss (aggression, farm-heavy, roaming).
  - Degrades gracefully to the crafted-feature base for cold-start players.
  - Requires corpus density per puuid to actually train — players with few games get mostly-zero residuals and fall back to features.

### Stream 3 — Dynamic sequence (hybrid time)

Per-game sequence over the match timeline. Two token types, interleaved at true time:

- **Frame anchor tokens** — emitted every 60s (Riot's frame cadence). Each carries full per-participant state: gold, XP, level, position, CS, jungle CS, for all 10 participants. Acts as a "state reset" so the sequence model doesn't have to reconstruct full state from events alone.
- **Event tokens** — one per meaningful event, at its millisecond timestamp. Event types: CHAMPION_KILL, BUILDING_KILL, ELITE_MONSTER_KILL, ITEM_PURCHASED, SKILL_LEVEL_UP, WARD_PLACED, WARD_KILL, CHAMPION_SPECIAL_KILL, plus inferred: RECALL, ENGAGE, DISENGAGE. Each tagged with actor participant (and target, where applicable).

Positional encoding: continuous in milliseconds from game start, not integer step index.

Approximate sequence length per game: ~35 anchors + ~200–400 events = ~250–450 tokens.

## Action space — two-tier

The "action-conditioned" principle requires clean separation between decisions and outcomes.

**Decisions** (what a participant chose):
- `ITEM_PURCHASED` — explicit from Riot timeline.
- `SKILL_LEVEL_UP` — explicit.
- `WARD_PLACED` (and `CONTROL_WARD_PLACED`) — explicit.
- `RECALL` — **inferred** from position jump to fountain + HP/mana reset at the minute-boundary.
- `ENGAGE` / `DISENGAGE` — **inferred** at fight starts from position clustering and first-blood-of-fight timing. Coarse; per-team initially, refined to per-participant later if signal warrants.

**Outcomes** (consequences the world predicts):
- `CHAMPION_KILL`, `BUILDING_KILL`, `ELITE_MONSTER_KILL`, `CHAMPION_SPECIAL_KILL` — world events downstream of decisions and prior state.

**Factorization**: the model predicts `p(outcome | state, decisions)`. This is the shape a teachable moment lives in: "players who recalled here lost the drake; players who stayed won it." If outcomes and decisions are conflated (one-tier), the retrieval layer has to reconstruct this factorization anyway — cheaper to bake it in.

**Inference cost caveat:** recall detection is straightforward; engage/disengage detection starts coarse (team-level, from position clustering + fight-start flags). Refinements are a follow-up.

## Architecture — RSSM world model

**RSSM-style (Dreamer-lineage) recurrent state-space model.**

### Why RSSM over a plain transformer

The decisive argument is **imagination rollouts** for counterfactual balance simulation (Milestone 6+). A supervised transformer can't cleanly answer "what happens across our corpus if IE crit is 30%?" — teacher-forcing real events breaks under counterfactual patch parameters. RSSM can: intervene on the latent / conditioning parameters, roll out in latent space, measure outcome shifts.

For standard supervised prediction + retrieval (Milestones 1–5), a transformer would be simpler. The balance-simulation use case is what earns RSSM's complexity.

### Components

- **Deterministic recurrent state** `h_t` (GRU).
- **Stochastic latent** `z_t` — sampled from posterior at each anchor step.
- **Prior network** `p(z_t | h_t)` — predicts latent before seeing the next observation (used for imagination rollouts).
- **Posterior network** `q(z_t | h_t, o_t)` — refines latent given the next observation (used during supervised training).
- **KL-balanced loss** between prior and posterior, with free-bits to prevent collapse.

### Wiring

- Each participant in the dynamic stream carries a per-participant state vector. Stream-2 player embedding is **fused** into that vector (concat + project) so "this participant at minute 12" always carries who they are.
- Stream-1 static context is produced once per game and **cross-attended** into the RSSM at every anchor step.
- Event tokens feed the RSSM between anchors; anchor frames provide the "observation" that drives the posterior update.

### Decoder heads (all four, off `z_t`)

All heads are distributional.

1. **Next meaningful event distribution** — categorical over (event type × actor × target). Largest loss weight. Trains `z_t` to carry event-predictive information.
2. **Next frame features** — regression head, distributional (μ, σ). Participant gold/XP/position deltas at the next anchor. Grounds `z_t` in observable state.
3. **Final outcome** — binary win/loss for the game. Sparse signal but causally critical. Forces `z_t` to carry victory-predictive information, which is exactly what retrieval needs.
4. **Next decision distribution** (per participant) — what each player chooses next. Ties the decision stream into the world model and enables decision-level counterfactuals at retrieval time.

**Deferred:** reward / value head. Only needed for policy learning (out of scope). LoL has no clean dense reward signal and post-hoc "reward" engineering is fragile.

**Loss weights:** next-event largest (95% target lives here). Outcome second (causal anchor). Frame and decision grounding. KL regularizer on `z_t` (standard RSSM).

## Milestones

Ladder of concrete deliverables. Each produces working, evaluable output.

### 1. Pipeline shakedown

Tokenizer + dataloaders + three-stream encoders wired end-to-end. A tiny causal transformer overfits a 50-game corpus to near-zero loss in minutes. Proves the pipeline carries signal and has no silent bugs.

### 2. Baseline classifier

Plain causal transformer (no RSSM) over the same three-stream input. Trains to predict the next meaningful event at each anchor boundary.

- **Target:** next-meaningful-event **top-5 accuracy ≥ 95%** on held-out games.
- **Framing:** stretch target, best-effort. Prior art (Honor of Kings event prediction) suggests this is steep. If we come up short at 80-90% top-5, that is still a strong result and a useful baseline.
- **Purpose:** validates the data carries the signal; establishes a ceiling RSSM must match or beat. An RSSM that underperforms this baseline is a bug signal.

### 3. RSSM v1

Full architecture from this document.

- Match or beat baseline top-5 on next-event head.
- **Outcome AUC ≥ 0.75** at minute 15 on held-out games.
- Frame-feature MSE reported on held-out set (no threshold — observability metric).
- Next-decision per-participant accuracy reported.

### 4. Retrieval check

For each held-out game and each anchor `t`, compute `z_t` and kNN the training corpus. For each retrieval, measure the binary-outcome (win/loss) entropy of the retrieved cohort.

- **Target:** mean cohort outcome entropy **≥ 0.7 bits** across held-out anchors (equivalent to retrieved cohort win-rates in roughly [0.2, 0.8] — not trivially decided).
- **Why this metric:** good retrieval finds games that split informatively (some wins, some losses from the same state). A cohort locked to one outcome carries no teachable signal; a cohort with a meaningful win/loss mix is where critical-point divergences live. Entropy is the principled one-number summary of "how split is this cohort."
- **Held-out anchor sampling:** restrict to mid-game anchors (minutes 10–25). Early-game and late-game anchors are either trivially 50/50 or trivially decided and don't stress retrieval.

### 5. First teaching surface

For each held-out game, identify the top-1 critical-point divergence at two polarities:

- **Mistake anchor (deficit lesson):** high-entropy cohort + held-out game on the minority **losing** side of the split. "This state splits; your branch is the losing one — here's what separated it."
- **Strength anchor (reinforcement lesson):** high-entropy cohort + held-out game on the minority **winning** side of the split. "This state splits; your branch is the winning one — here's what you did that most players in your position didn't."

Per game, generate one lesson of each polarity (when both exist) and let the downstream lesson-selector balance frequency based on player state. Strength lessons are not optional garnish; they are co-equal to mistake lessons by design.

- **Evaluation:** manual QA on 20 held-out games. Does each lesson (mistake and strength) correspond to a plausible decision that differentiated winners from losers at that anchor? QA must sample both polarities; a system that produces sharp mistake lessons but weak/generic strength lessons fails this milestone.
- This is when the product has a face.

### 6+ (future, out of this plan)

- Imagination rollouts for patch-balance sweeps. Intervene on the patch-parameter conditioning vector, roll out in latent space, measure aggregate outcome shifts. Deliverable: balance-simulation API.
- Refined engage/disengage decision inference.
- Spaced-repetition across lessons; per-player progression tracking.

## Open questions deferred to the implementation plan

- Exact `d_model`, `z_t` dimension, GRU hidden size — pin during pipeline shakedown based on overfit behavior.
- KL balance ratio and free-bits hyperparameters — RSSM-typical; tune during Milestone 3.
- How many events per game to cap for batching — measure during shakedown.
- How recall / engage / disengage inference is implemented concretely — separate small design inside the plan.
- kNN index choice (FAISS vs brute force at first) — Milestone 4 scope.

## Not revisiting

These are locked and should not be reopened at plan time without new evidence:

- Three-stream input shape.
- Two-tier decision/outcome split.
- RSSM over plain transformer (baseline transformer exists only as Milestone 2 yardstick).
- Patch as rich numeric conditioning, not categorical.
- All four decoder heads.
- Hybrid time (anchors + events), not per-minute-only or per-event-only.
