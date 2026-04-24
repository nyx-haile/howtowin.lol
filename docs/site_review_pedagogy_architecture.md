# Site Review — Pedagogy Architecture

Maps the 6-layer teaching stack in `docs/site_pedagogy_engagement_lit_review.md` onto the live route tree at `site/src/routes/review/preview/`. This doc is the bridge between the literature review and the code.

The scaffold is shipped against mock fixtures. When `9b01bf83-3b7` + `9b01bf83-8hi` land, only the fixture loader changes — the component tree, state machine, and pedagogy contracts stay intact.

## The six layers

| Layer | Purpose | Lit review § | Surface |
|---|---|---|---|
| 1 — Detection | Find high-entropy mid-game anchors | §3.1 | `code/model/lesson.py` (`LessonResult`) |
| 2 — Presentation | Pretest → elaborated feedback | §4.2, §4.3 | `LessonStage` · `PretestCard` · `RevealCard` |
| 3 — Intervention | Forethought-gap intent commit | §4.5 (Kleinman 2021) | `InterventionPrompt` · `IntentReceipt` · `streakStore` |
| 4 — Spaced reinforcement | Expanding intervals, interleaved themes | §4.1, §4.4 | `SpacedQueueStrip` · `InterleavedBadge` (stub) |
| 5 — Retrieval check | Testing effect — cued recall day-after | §4.4 (Roediger & Butler) | `RetrievalDrillCard` (stub) |
| 6 — Engagement scaffolding | Supportive, non-corrosive motivation | §4.6 (Renaud 2024) | `StreakBadge` · `WeeklyLeagueCard` · `SupportivePushPreview` · `ProgressiveDisclosureGate` |

## Route tree

```
site/src/routes/review/
  +page.svelte                      # live waitlist — UNTOUCHED
  preview/
    +layout.svelte                  # gold "PREVIEW · mock data" banner + noindex
    +page.svelte                    # fixture index (layer bridge: L1 list)
    [fixtureId]/
      +page.ts                      # load() picks fixture from registry
      +page.svelte                  # hosts LessonStage + L4–6 under disclosure gate
```

**Promotion path** (post-3b7): rename `preview/` away, swap fixture import for `queryPython({ action: 'lesson', ... })`, rename `[fixtureId]` → `[matchId]-[team]` to match the `/games/[matchId]-[playerId]/` convention already in use.

## State machine

`lessonSession.ts` — one writable store per fixture, re-initialized on route param change:

```
  pretest → reveal → intervention → committed
     │
     └── no-anchor   (terminal, renders LessonUnavailable)
```

Store is **not URL-encoded** (would leak pretest answers into browser history) and **not page-data** (can't update on interaction). Refresh resets to `pretest` — acceptable for v1 stakeholder walkthrough. Session resumption is a separate future bead.

`advanceToReveal` guards on a non-null `pretestChoice`. `commitIntent` is the only path that reaches `committed`.

## Epistemic contract

All cohort prose flows through `epistemic.ts:hedge(claim, confidence)`, with `confidence` derived from `anchor.cohort_entropy`:

| Entropy | Confidence | Prefix |
|---|---|---|
| < 0.3 | strong | "In most cohort games, " |
| 0.3–0.7 | moderate | "Cohort evidence suggests " |
| ≥ 0.7 | weak | "This signal is noisy, but " |

`adapter.ts:toLessonViewModel` is the **only** site-side function that wraps claims. When `9b01bf83-8hi` prose lands, it enters through the same adapter — no hand-authored string bypasses hedging. Banned phrases are listed in `docs/site_copy_guide.md`.

## Streak semantics

Streak advances **only** on `InterventionPrompt.accept(intent)` → `streakStore.recordIntervention(intent)`. This is the forethought-gap event (Kleinman 2021, lit review §4.5), explicitly not:

- opening the app (Duolingo pattern, flagged corrosive for coaching products)
- visiting `/review/preview`
- opening a fixture
- viewing a reveal
- selecting a pretest answer

The rule is enforceable because `streakStore` has a single write path and no other caller.

localStorage keys: `hw:streak:count`, `hw:streak:lastMeaningfulISO`, `hw:streak:commits` (ring buffer, last 30).

Degenerate-mode guardrail: if a user closes a session after reveal but before intervention, a soft toast on the next visit offers "you saw the lesson — want to commit an intent?" (filed as follow-up; plan §Streak semantics).

Prod guardrail — streak verification via real next-game ingress — is filed as `9b01bf83-109`, blocked on Riot prod API access.

## Progressive disclosure

`ProgressiveDisclosureGate` hides Layers 4–6 until `onboardingStore.lessonsCompleted >= requiredLessons` (default 1). Rationale: stabilize the core pretest → reveal → intent loop before adding queue/drill/league surfaces. Lit review §4.6: "layer surfaces only once the habit underneath is stable."

## Scaffold fading (deferred)

All components that will eventually fade on expertise accept a `scaffoldLevel: 0 | 1 | 2` prop, hardcoded to `0` in v1. Reserves the prop so v2 fading (filed as `9b01bf83-15p`) is not a breaking change.

## Schema sync

Lesson dataclasses live in `code/model/lesson.py`. The site mirrors them via three artifacts:

1. `site/src/lib/lesson/schema.ts` — hand-written TS, documented with `@see`.
2. `site/src/lib/lesson/schema.lock.json` — exported from `code/model/export_lesson_schema.py` via `dataclasses.fields()`, checked into git.
3. `site/scripts/check-lesson-schema.ts` — asserts every locked field appears in `schema.ts`. Runs under `bun run check`.

Drift policy: when `lesson.py` changes, re-export the lock. CI fails if lock is stale or TS is missing a field.

## Non-goals (explicit)

- **Archetype labels in user-facing copy** — per `9b01bf83-8hw`. Cohort retrieval stays action-basis ($z_t$). Archetype-conditioned prose substitutions, if the design lands, enter only through `adapter.ts` — never as cohort filters or user-visible labels.
- **Open-app streaks** — lit review flags as corrosive for coaching.
- **Causal copy** — no "you would have won", "this will win you". See `docs/site_copy_guide.md`.
- **Real lesson CLI wiring** — blocked on `9b01bf83-4tl` + `9b01bf83-8hi`. Implementation bead is `9b01bf83-6h7`.
- **Thesis Labs sticker** — blocked on outreach (`9b01bf83-aa9`), per user directive.
- **Scaffold fading v1** — `scaffoldLevel` prop is reserved, behavior is `9b01bf83-15p`.
- **Custom ESLint rule for banned phrases** — `9b01bf83-9ip`.
- **Session resumption across refresh** — v1 is a stakeholder walkthrough, not prod.
- **FastAPI sidecar** — `9b01bf83-oaj`, still "needs design" (subprocess-on-host is documented as interim).

## Reuse

- `site/src/lib/ProgressBar.svelte` — used by `CohortSparkline`, `WeeklyLeagueCard`.
- `site/src/lib/ConceptBadge.svelte` — reusable; extend `COLORS` when new themes are introduced.
- `site/src/lib/MomentCard.svelte` — **wrap, do not modify**. `AnchorMomentCard.svelte` composes the anchor shape; legacy `/games/` code path is unaffected.

## Verification

Dev loop (`bun run dev` inside `site/`):

- `/review/preview` — fixture index.
- `/review/preview/loss_mid_mistake_14m` — pretest → reveal → intervention → streak=1.
- `/review/preview/win_late_strength_22m` — strength polarity copy, streak=2.
- `/review/preview/nothing_to_teach_early_stomp` — `LessonUnavailable`, streak unchanged.
- `/review/preview/loss_prose_missing_ambiguous` — reveal renders with fallback prose.

Regression checks (enforced by inspection):

1. `/review` still shows the waitlist form.
2. No archetype label renders anywhere.
3. Streak does NOT advance on app open, route visit, fixture open, reveal view.
4. Streak advances only on `InterventionPrompt.accept`.
5. Refresh mid-flow → phase resets to `pretest`.
6. Spot-check fixtures for unhedged "will" / "would have".

Automated gates: `bun run check` (svelte-check + `check:schema`) must pass.
