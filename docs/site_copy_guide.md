# Site copy guide — pedagogy surfaces

Short, enforceable rules for any user-facing string under `site/src/lib/lesson/` or the routes that render it. Rationale lives in `docs/site_pedagogy_engagement_lit_review.md`.

## The hedge rule

Every claim about cohort outcomes routes through `hedge(claim, confidence)` in `site/src/lib/lesson/epistemic.ts`. Raw claims go in ("the fight goes your way"); hedged sentences come out. Do not hand-hedge — the helper picks the prefix from `cohort_entropy` so the strength of the language tracks the strength of the evidence.

If you find yourself writing `"you likely would have ..."` directly in a Svelte template, stop. Route it through `hedge()` or the `LessonViewModel` adapter.

## Banned phrases → replacement patterns

| Banned | Why | Use instead |
|---|---|---|
| `you will ...` | Predictive certainty we can't back | `this tends to correlate with ...` |
| `this would have won the game` | Unhedged counterfactual | `cohort evidence suggests ...` (via `hedge`) |
| `you should always ...` | Prescriptive; ignores context | `most winners in your cohort ...` |
| `you're a [Archetype]` | Archetypes are latent (bead 9b01bf83-8hw) | Describe the behavior directly: `your first gank averages 5:30 vs cohort 3:45` |
| `X% chance of winning` | Readers promote percentages to facts | `X of Y cohort games ended in a win` (see `cohortOutcomeLabel`) |
| `your KDA is bad` | Shames; also descriptive-not-causal | Name the specific decision + moment |
| `correct answer` on pretest reveal | Pretest is retrieval practice, not a quiz | `what cohort winners usually chose` |

## Tone

- **Supportive, not scolding.** Players in a losing streak reach for coaching at their most emotionally-flooded moment. The lit review (§Risks item 5) warns that piling "here's what you did wrong" on a tilted user is the self-sabotage failure mode.
- **Concise.** Shute 2007 (lit review §4.6): short elaborated feedback beats long. Three lines max on reveal copy.
- **Behavioral, not identity.** "Your wards at minute 6 cluster near fountain" — not "you're a passive warder".

## Copy around streaks

The streak advances on intervention commit, not on app-open. Reflect that in the UI text:

- On commit: `intent locked in — we'll check in after your next match` (future-tense "check-in", not retroactive "good job").
- On soft re-prompt: `you saw the lesson — want to commit an intent?` (no guilt, low friction).
- Never: `don't break your streak!` (loss-aversion pattern the lit review flags as corrosive in a coaching product).

## Push copy (L6 stub previews)

`SupportivePushPreview.svelte` cycles through variants. All must:

- Start with a specific callback to the player's own last session ("about that mid-game recall...").
- Avoid superlatives and exclamation points.
- Never threaten streak loss.

## Audit

Spot-check before PR: grep the preview routes for the banned phrase list. No hits.

```
rg -n "you will|would have|should always|correct answer" site/src/routes/review/preview site/src/lib/lesson
```

Hits in `epistemic.ts`, `schema.ts`, and this guide are fine (they're the rule source). Hits anywhere else need either (a) rewording or (b) routing through `hedge()`.
