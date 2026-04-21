# Model audit vs notes + `docs/superpowers`

Date: 2026-04-21
Related bead: `9b01bf83-pfw`

## Scope I reviewed

- `code/model/*`
- `code/parser.py`, `code/features.py`, `code/concepts.py`
- `docs/superpowers/specs/*`
- `docs/superpowers/plans/*`
- `docs/*eval_report*`
- `notes/*`

## Quick state of the repo

- Data scale in `data/howtowin.db`: **12,846 games**, **3,654,210 frames**, **67,188,501 events**, **20,377 distinct players**.
- Current splits: **12,745 train / 64 game-cold / 37 player-cold**.
- Most games are concentrated on two patches (`16.7.760.9485`, `16.8.764.3737`), so patch-only conditioning is weak unless richer numeric patch features are actually enabled.

## What the current model actually is

### Implemented path

1. **Ingestion / storage**
   - `code/parser.py` stores raw Riot match + timeline in `raw_matches.db`.
   - It also writes a reduced relational view into `games`, `frames`, `events`, `players`.

2. **Plan A baseline**
   - `code/model/baseline.py` is a causal transformer over three streams.
   - The target is a **multi-hot next-minute event-type set**, not a full next-event / actor / target distribution.

3. **Plan B model**
   - `code/model/plan_b_model.py` is an RSSM-like anchor-level model:
     - static encoder -> `h0`
     - player encoder -> added to dynamic token embeddings
     - event tokens between anchors -> **mean-pooled** into one `action_summary` per anchor
     - frame features -> projected into one `obs` vector per anchor
     - RSSM steps **once per anchor**, not once per event token
   - Heads predict:
     - next event **type** set
     - outcome
     - coarse next decision class
     - next frame deltas on a small feature subset

4. **Retrieval / lesson MVP**
   - `code/model/retrieval.py` builds a kNN index over `[h_t || post_mu_t]`.
   - `code/model/m4_eval.py` computes cohort entropy.
   - `code/model/lesson.py` picks high-entropy anchors, but does **not yet explain what decision separated winners from losers**.

## What is already solid

- The repo now has a coherent end-to-end modeling stack instead of only the older feature/concept pipeline.
- Leak discipline is much better than the early Plan A state:
  - `exclude_match_ids` plumbing exists
  - player-cold split exists
  - frozen-minute-0 leak probe exists
- Raw timeline blobs are preserved in `raw_matches.db`, which means richer featurization can be added later without re-fetching matches.
- `code/features.py` and `code/concepts.py` already contain useful macro/game-state ideas that can be reused as auxiliary targets or lesson scaffolding.

## Biggest gaps against the spec/plans

### 1) The **static game-context stream is mostly unimplemented**

This is the biggest architectural gap.

The sequence-model spec expects the static stream to include:
- 10 champion picks
- sides
- queue / region / time-of-day
- rich numeric patch conditioning
- cross-attention into the model at every anchor

What the code currently does instead:
- `MatchDataset` calls `patch_vector_for_match(mid)` with the default scaffolding mode
- in scaffolding mode the vector is effectively just the **patch version triple**
- `PlanBModel` uses static context only to seed `h0` via `static_to_h`
- there is **no static cross-attention at each anchor step**
- champion IDs / side / queue / region / time-of-day are not part of the model input path

So the model currently has far less pre-game context than the spec assumes.

**Tracked follow-up:** `9b01bf83-3h8`

### 2) The dynamic stream keeps the timeline shape, but then collapses it too aggressively

The spec describes a hybrid-time model where event tokens matter individually between anchors.

Current code simplifications:
- `tokenizer.py` emits event tokens at fine timestamps
- but `PlanBModel` does **not** run RSSM updates over those event tokens
- instead it mean-pools all events in a minute window into one `action_summary`
- the RSSM advances once per anchor only

This loses:
- within-minute event ordering
- burst structure of fights/objectives
- separation between multiple decisions in the same window

This is probably one reason the current model feels more like an **anchor-level sequence classifier with pooled context** than a true world model.

### 3) Event semantics are much coarser than the spec wants

The code records more detail than the model uses.

Examples from `events.details`:
- `ITEM_PURCHASED` has `itemId`
- `SKILL_LEVEL_UP` has `skillSlot`
- `ELITE_MONSTER_KILL` has `monsterType` / `monsterSubType`
- `BUILDING_KILL` has `buildingType` / `laneType` / `towerType`

But the modeling path currently ignores all of that:
- token vocab is only 11 coarse event types
- `target_slot` is stored in the tokenizer token but dropped before modeling
- no target embedding exists in `dataset.py` / `encoders.py`
- event head predicts only event **type**, not event type × actor × target × subtype

That means the model cannot distinguish:
- Boots purchase vs Infinity Edge purchase
- Q level-up vs R level-up
- dragon vs Baron vs Herald cleanly in the token stream
- tower kill top vs mid vs inhibitor tower

This is a major limitation for both counterfactual advice and later lesson generation.

**Tracked follow-up:** `9b01bf83-kzx`

### 4) Anchor observations are too thin relative to both the notes and the spec

The current anchor observation is only **10 × 6** features:
- `total_gold, xp, level, pos_x, pos_y, cs`

That is much smaller than what the notes point toward (`notes/timeline_architecture`, `notes/clickhouse_array`) and smaller than what the repo already knows how to derive.

Missing or underused state includes:
- `current_gold`
- `jungle_cs`
- `kills/deaths/assists`
- ward / item state
- explicit objective state
- team-level macro deltas (gold diff, tower diff, objective counts, centroid/spread)
- richer combat/resource stats from the raw timeline structure

Important detail: `code/features.py` already computes several strong macro features, but the Plan B model does not consume them.

This is likely one of the highest-leverage improvements because it aligns with both:
- the original notes' emphasis on positional / objective / differential state
- the current teaching goal of "same state, different branch"

**Tracked follow-up:** `9b01bf83-3iv`

### 5) The RSSM implementation is useful, but still notably lighter than the Plan B design

Main divergences:
- no Dreamer-style KL-balanced stop-gradient formulation; current code uses a simpler free-bits KL term
- decoder heads read only `z_t`, while the M4 design rationale assumes the heads consume `[h_t || z_t]`
- rollout auxiliary loss uses the **last observed labels for every imagined step** instead of true future-step labels
- `imagination_rollout_top5()` seeds `h` from `static_to_h(static)` instead of the true saved `h_t`

This makes the rollout story weaker than the spec/plans imply.

The weird eval curve in `docs/plan_b_eval_report_2026-04-17.md` (rollout top-5 increasing with step instead of decaying) is a symptom that the rollout eval/training path is still approximate.

**Tracked follow-up:** `9b01bf83-2ew`

### 6) Retrieval passes the gate, but the teaching surface is still only half-built

Current state:
- M4 retrieval passed the entropy gate
- `lesson.py` can pick a high-entropy anchor
- but it still does not answer the core teaching question:
  - **what did winners do differently from losers in this cohort?**

Also, the retrieval report shows the already-filed follow-up:
- `howtowin.lol-bhh` — inverted / front-loaded entropy curve by minute

And the lesson layer still needs the already-filed work:
- `9b01bf83-8hi` — lesson content from cohort vs target decision diff
- `9b01bf83-8hw` — latent archetypes for lesson-plan personalization

So the latent/retrieval core is present, but the actual Duolingo-style teaching layer is still at MVP-anchor-selection stage.

## Best alignment with the notes

The old notes are not RSSM docs, but they do point to a few durable ideas that still fit the current roadmap:

### A. Notes strongly favor **axis-wise timeline tensors**

`notes/timeline_architecture` and `notes/clickhouse_array` both push toward a richer time × player × stat representation.

Current repo status:
- good news: the parser already stores enough structure to move in that direction
- bad news: the model currently compresses that down to a very small anchor observation

So the right move is **not** a redesign; it is to widen the observation/state bundle and token semantics.

### B. Notes care about **position, objectives, and derived differentials**

`notes/24.5.15` explicitly calls out things like:
- weighted gold diff
- positional evidence (e.g. jungler in bot river at 5 min)
- correlation of stats with winrate

That maps cleanly onto:
- adding macro/team-diff auxiliary heads
- reusing `features.py` outputs as side-channel supervision
- giving the model better objective/position state at each anchor

### C. Notes worry about **patch drift / stale derived storage**

`notes/24.8.8` and `notes/24.8.12` prefer keeping raw match IDs / raw data because patches change.

Current stack actually supports this well:
- raw match + timeline blobs are stored
- patch vector scaffolding avoided a leak

The next step is to use that safety correctly by rebuilding richer features from raw blobs / Data Dragon, rather than staying stuck at a near-empty static vector.

## Highest-leverage improvement order

If I were sequencing work from here, I would do it in this order:

1. **Fix input fidelity before adding more fancy downstream logic**
   - static stream completion (`9b01bf83-3h8`)
   - richer anchor observations (`9b01bf83-3iv`)
   - target/subtype-aware event tokens (`9b01bf83-kzx`)

2. **Tighten the world-model mechanics**
   - true rollout seeding / future labels (`9b01bf83-2ew`)
   - optionally move heads to consume `[h_t || z_t]` if that remains the retrieval contract
   - consider a cleaner action/outcome factorization instead of pooled event windows

3. **Only then push harder on teaching-surface work**
   - cohort-vs-target decision diff (`9b01bf83-8hi`)
   - archetype/personality routing for lesson framing (`9b01bf83-8hw`)

4. **Then productionize retrieval**
   - FAISS / serving concerns (`howtowin.lol-l2z`)

## Concrete thesis

The repo already has the skeleton of the intended system:
- leak-aware world-model training
- latent retrieval
- first-pass lesson-anchor selection

But right now the model is bottlenecked less by "needing a smarter architecture" and more by **underpowered state representation**:
- too little static context
- too little per-anchor observation detail
- too little event semantics
- too much pooling / approximation in the rollout path

That is also the main place where the current code diverges from both:
- the original notes (rich per-axis timeline state)
- the `docs/superpowers` design (real static context, richer action semantics, stronger rollout contract)

In short: **the next big win is to make the current model see more of the game you already store, before inventing a more complex model on top of the compressed view.**
