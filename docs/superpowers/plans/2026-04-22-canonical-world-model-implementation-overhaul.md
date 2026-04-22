# Canonical World-Model Implementation Overhaul Plan

**Date:** 2026-04-22
**Status:** Active planning doc for the first architecture-alignment implementation pass.
**Primary spec:** `docs/superpowers/specs/2026-04-21-canonical-world-model-spec.md`

## Purpose

This plan translates the canonical world-model spec into a bead graph that is safe to implement against.

Goals:

1. make the next implementation steps explicit;
2. ensure the first aligned retrain has a well-defined scope;
3. keep temporary proxies from silently masquerading as canonical compliance;
4. shape the beads dependency graph so the active work surface matches the current phase.

**Queue note:** the installed `bd` version exposes `bd ready`, not `bd next`. This plan treats `bd ready` as the operative "next work" surface, and structures dependencies so the top ready work is the right implementation surface for the current phase.

## Bead audit against the canonical spec

### Already aligned with the canonical spec direction

These beads remain valid and should stay in the overhaul stack:

- `9b01bf83-3h8` — fill out the static game-context feature bundle.
- `9b01bf83-3iv` — widen anchor observations with richer participant/team/objective state.
- `9b01bf83-kzx` — make event tokens and heads target-aware / subtype-aware.
- `9b01bf83-2ew` — replace approximate rollouts with true latent rollouts.
- `9b01bf83-8hi` — downstream lesson content from cohort vs target decision differences.
- `9b01bf83-8hw` — downstream lesson-plan personalization design.
- `howtowin.lol-bhh`, `howtowin.lol-cow`, `howtowin.lol-rzz` — retrieval diagnostics that remain useful after the first aligned checkpoint exists.

### Missing from the current bead graph

The canonical spec added or clarified requirements that did **not** yet have dedicated implementation beads:

1. **Per-step static conditioning** — the spec requires Stream 1 to condition every step, not just seed `h0`.
2. **Event-token recurrence between anchors** — pooled minute summaries are still a proxy.
3. **Head/input contract alignment** — decoder heads and retrieval should share `r_t = [h_t || z_t]`.
4. **First aligned retrain + eval pass** — once the architectural deltas land, the project needs a dedicated retrain/eval bead rather than treating retraining as implicit.
5. **Aligned retrieval re-run** — M4 and its diagnostics need to be rerun on the aligned checkpoint, not left attached to the pre-overhaul model.

### No longer on the critical path for this overhaul

These issues may still matter, but they should not drive the immediate implementation queue for canonical compliance:

- `howtowin.lol-046` — historical Plan A baseline retrain after leak fix.
- `9b01bf83-6c5` — scale corpus to 50k games.
- product/deployment tasks such as `9b01bf83-ncj`, `9b01bf83-6l7`, `9b01bf83-3b7`, and the ThesisLabs work.

## Decisions for the first architecture-alignment pass

These answer the five open questions named in the canonical spec.

### 1. Anchor observation schema

The first full-fidelity anchor bundle should include:

- **Per participant:** total gold, current gold, xp, level, CS, jungle CS, kills, deaths, assists, position x/y, and available frame-native health/power summaries.
- **Per team / objective macro:** gold diff, xp diff, kill diff, tower diff, plate diff, neutral-objective counts/state (drakes, grubs, Herald, Baron, Elder), and team centroid/spread summaries where already derivable.
- **Scope cut for the first pass:** do **not** block the first aligned retrain on reconstructing a perfect per-frame full inventory vector if the event stream already carries item identity. Inventory/warding summaries are allowed where easy, but full event-semantic fidelity belongs primarily in the event stream.

### 2. Event target factorization

Use a **factorized next-event objective**, not a single giant flat categorical.

For the first aligned pass, event modeling should include:

- coarse event type;
- actor slot;
- target slot when applicable;
- subtype / payload heads gated by event type (item id, skill slot, monster subtype, building subtype, ward subtype, etc.).

This keeps the target canonical-compliant without exploding a flat class count.

### 3. Static conditioning mechanism

Adopt **tokenized static context with per-step cross-attention** in the first aligned pass.

Concretely:

- represent the static stream as a small token set (champion/side tokens plus a misc-context token for queue, region, time bucket, patch parameters, etc.);
- condition every dynamic update step on those tokens;
- do **not** treat `h0 = f(static)` alone as compliant Stream-1 conditioning.

### 4. Recurrence staging plan

Go directly to **event-token recurrence between anchors** in the first aligned pass.

Rationale:

- the pooled action-summary proxy is one of the main remaining architectural mismatches;
- removing it after another retrain would create avoidable churn;
- the canonical spec explicitly calls out hybrid time as locked.

### 5. Head/input alignment

Move all four core decoder heads to the shared anchor representation:

```text
r_t = [ h_t || z_t ]
```

Retrieval keeps using `[h_t || μ_q(z_t)]`, so training and downstream retrieval should finally agree on what the anchor representation is.

## Implementation phases

### Phase 0 — planning / queue shaping

Primary bead:

- `9b01bf83-znx` — this plan bead.

Required result:

- decisions above recorded;
- missing beads added;
- dependency graph updated so `bd ready` shows the right surface.

### Phase 1 — canonical input/state fidelity

These are the first implementation beads after planning:

1. `9b01bf83-3h8` — static game-context feature bundle.
2. `9b01bf83-3iv` — richer anchor observations.
3. `9b01bf83-kzx` — target-aware / subtype-aware event semantics.
4. **new bead:** per-step static conditioning.

These establish the canonical three-stream inputs.

### Phase 2 — canonical sequence-model mechanics

After Phase 1:

5. **new bead:** replace pooled between-anchor action summaries with event-token recurrence.
6. **new bead:** align decoder heads on `[h_t || z_t]`.
7. `9b01bf83-2ew` — make rollouts seed from true latent state and score against true future labels.

These establish the canonical recurrence / head / rollout contract.

### Phase 3 — aligned retrain and re-eval

After Phases 1–2:

8. **new bead:** train the first canonical-alignment checkpoint and rerun Plan B-style evals (game-cold, player-cold, leak probe, rollout diagnostics).
9. **new bead:** rebuild retrieval index and rerun M4 on the aligned checkpoint, including baseline contrast.

### Phase 4 — downstream teaching surfaces

Only after the aligned checkpoint and retrieval rerun exist:

10. `9b01bf83-8hi` — lesson content from cohort/target decision diff.
11. `9b01bf83-8hw` — archetype/personalization design.
12. `howtowin.lol-bhh`, `howtowin.lol-cow`, `howtowin.lol-rzz` — retrieval diagnostics against the aligned checkpoint.

### Phase 5 — scale and productization

After the first aligned checkpoint is understood:

13. `9b01bf83-6c5` — scale corpus to 50k.
14. `howtowin.lol-l2z` — FAISS / production retrieval.
15. site/backend/product tasks.

## Bead-graph rules

To keep the queue honest:

- no architecture-alignment implementation bead should be ready before `9b01bf83-znx` is closed;
- rollout work must not become ready until input/state-fidelity and recurrence/head-alignment work land;
- teaching-surface and retrieval-diagnostic beads should depend on the first aligned retrain/re-eval, not only on the older pre-overhaul checkpoint;
- corpus-scale work should wait until the first aligned architecture is stable enough to deserve a large retrain.

## Exit condition for this plan

This planning pass is complete when:

1. the decisions in this doc are reflected in the bead graph;
2. the missing implementation beads exist;
3. stale/non-critical work is no longer crowding the top of the active implementation queue; and
4. the next ready work after this plan closes is the Phase 1 compliance work, not legacy proxy work.
