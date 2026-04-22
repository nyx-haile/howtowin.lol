# ThesisLabs Integration Brief (Internal Draft)

**Date:** 2026-04-22  
**Status:** Internal planning artifact for bead `9b01bf83-92t`. No outreach has been sent.

## Why this brief exists

The repo now has a canonical world-model spec and a concrete implementation-overhaul plan, but the first architecture-alignment retrain has **not** happened yet. That makes this the highest-leverage point for outside algorithm review: the project has already shown proof-of-life signal, but the next retrain will harden several architectural choices that are still worth pressure-testing.

## Current repo/spec state in one page

### Locked upstream direction

The canonical docs now lock these points:

- the product goal is a **League of Legends world model for post-game teaching**, not a dashboard;
- positive and negative lessons are co-equal;
- the model must use **three streams** (static game context, player priors, dynamic sequence);
- **hybrid time** is required (anchor frames plus event tokens);
- **decision/outcome separation** is required;
- RSSM-style latent dynamics remain the target family.

Primary references:

- `docs/superpowers/specs/2026-04-21-canonical-world-model-spec.md`
- `docs/superpowers/plans/2026-04-22-canonical-world-model-implementation-overhaul.md`

### What the current implementation still does the old way

The current Plan B code is still on the pre-overhaul approximation path:

- `code/model/plan_b_model.py` uses a thin `10 x 6` anchor frame bundle (`FRAME_FEAT_DIM = 6`);
- static context mainly seeds `h0` via `static_to_h`, rather than conditioning every step;
- between-anchor dynamics are pooled through `ActionSummarizer`, not true event-token recurrence;
- decoder heads read from `z_t` only;
- retrieval in `code/model/retrieval.py` already uses `[h_t || μ_q(z_t)]`, so training and retrieval are currently misaligned.

### What the latest evidence says

The latest committed reports are still **pre-overhaul**:

- `docs/plan_b_eval_report_2026-04-17.md` shows real outcome signal and a clean leak probe, but it does **not** yet satisfy the stricter monotonic-AUC / rollout-ordering expectations.
- `docs/m4_retrieval_eval_report_2026-04-17.md` passes the entropy gate, but the model still trails the hand-crafted frame-feature baseline and shows a front-loaded mid-game entropy curve.

Implication: external help is most useful if it sharpens the **next architecture pass**, not if it re-litigates whether any signal exists at all.

### Where outside input would change the queue

The overhaul plan says the next meaningful implementation surface is:

- **Phase 1:** richer static bundle (`9b01bf83-3h8`), richer anchors (`9b01bf83-3iv`), target-aware event semantics (`9b01bf83-kzx`), and a new per-step static-conditioning bead;
- **Phase 2:** event-token recurrence, head/input alignment on `[h_t || z_t]`, and true rollout seeding/scoring.

So the most valuable ThesisLabs input is the kind that reduces design churn **before** the first aligned retrain, not generic advice that lands after those choices are already implemented.

## Top algorithm questions to put in front of ThesisLabs

These are the highest-value questions from the canonical plan/spec to pressure-test before the first aligned retrain.

| Topic | Current first-pass direction | What we want reviewed |
|---|---|---|
| **1. Anchor observation schema** | First aligned pass should widen anchors to richer participant/team/objective state. | What is the smallest anchor bundle that is still a fair test of state fidelity, and which features are most important to add first (especially inventory/warding, objective state, and spatial summaries)? |
| **2. Event target factorization** | Use factorized next-event heads: event type + actor + target + gated subtype/payload heads. | Is this the right decomposition for sparse LoL event semantics, or should some payloads be folded into shared hierarchies / conditional heads to reduce class imbalance and training brittleness? |
| **3. Per-step static conditioning** | Tokenized static context with per-step cross-attention. | Is cross-attention the right conditioning mechanism for this model/data scale, or would FiLM/gating/hybrid conditioning preserve the needed patch + draft information with less complexity or instability? |
| **4. Between-anchor recurrence** | Skip another pooled proxy pass and go straight to event-token recurrence. | What recurrence pattern is most defensible here: per-event updates, chunked event blocks, or another hybrid that preserves time order without making training too fragile on modest hardware? |
| **5. Head / retrieval representation alignment** | Move all core heads onto `r_t = [h_t || z_t]`, matching retrieval's `[h_t || μ_q(z_t)]`. | Is one shared anchor representation the best contract for both prediction and retrieval, and what regularization / ablations are needed to verify the model is not just moving complexity around? |

### Secondary question if time permits

- **Evaluation design for the first aligned retrain:** which ablations and diagnostics are the minimum necessary to tell whether the canonical pass actually improved teachable state structure rather than only headline AUC?

## Artifact/context pack to share

Start with a **docs-first packet**, then add code pointers only if the conversation gets concrete.

### Must-share first

1. `docs/superpowers/specs/2026-04-21-canonical-world-model-spec.md`  
   Canonical source of truth for what is locked, what is proxy-only, and what remains open.
2. `docs/superpowers/plans/2026-04-22-canonical-world-model-implementation-overhaul.md`  
   Shows the actual implementation queue and the first-pass decisions already made.
3. `docs/plan_b_eval_report_2026-04-17.md`  
   Proof-of-life evidence: outcome signal, leak probe, and rollout caveats.
4. `docs/m4_retrieval_eval_report_2026-04-17.md`  
   Shows the retrieval gap that still needs architectural improvement.

### Share next if they want implementation reality, not just spec intent

- `code/model/plan_b_model.py`
- `code/model/rssm.py`
- `code/model/heads.py`
- `code/model/retrieval.py`
- `code/model/encoders.py`
- `code/model/dataset.py`
- `code/model/player_features.py`

These are the files that reveal the actual approximation choices in flight.

### Optional background only

- `docs/superpowers/specs/2026-04-15-sequence-model-design.md`

Use this only if they want the original product/teaching rationale or historical context behind the world-model direction.

### Do not lead with this

- raw `data/` artifacts;
- full database dumps;
- large checkpoint handoffs;
- broad product/site context.

The first discussion should stay focused on architecture choices and eval strategy.

## Proposed collaboration modes

1. **Async design red-team (best first step)**  
   Send the docs packet and ask for a written memo answering the five algorithm questions above, including recommended alternatives and failure modes.
2. **60-90 minute architecture review**  
   Walk through current proxy vs canonical target, then force decisions on the highest-risk contract points: anchor schema, conditioning path, event recurrence, and representation alignment.
3. **Post-retrain checkpoint review**  
   After the first aligned retrain lands, review eval outputs and retrieval baselines to decide whether the architecture is genuinely better or just differently optimized.

## Recommended first-contact agenda

If/when outreach happens, the first meeting should stay tightly scoped:

1. **5 min — product frame**  
   "We are building a LoL world model for post-game teaching, not a generic analytics dashboard."
2. **10 min — current state snapshot**  
   Canonical spec is set; current code still uses thin proxies; proof-of-life exists; retrieval is still weaker than the frame-feature baseline.
3. **25 min — the five algorithm questions**  
   Work from anchor schema → event factorization → static conditioning → event recurrence → head/retrieval alignment.
4. **10 min — recommended collaboration shape**  
   Decide whether the next deliverable should be an async memo, a proposed ablation plan, or an implementation review.
5. **5 min — immediate next action**  
   Leave with one explicit artifact request (for example: "send back a ranked recommendation on conditioning + recurrence before the first aligned retrain").

## Internal prep before any outreach

- turn the five algorithm questions above into a one-page prompt;
- pull 1-2 diagrams/screenshots from the canonical spec or model code so the current-vs-target gap is visible immediately;
- be explicit that the current goal is **pre-retrain architecture review**, not a generic product brainstorm.

That framing keeps the collaboration narrow enough to be useful.
