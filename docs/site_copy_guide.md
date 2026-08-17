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

The banned-phrase table is enforced mechanically. `site/eslint-rules/no-epistemic-overclaim.js`
is a local ESLint rule that transcribes the table above (plus the two hard "never" lines from
§Copy around streaks and §Push copy) and fails the build on a hit.

```
cd site && bun run check:epistemic-rule
```

That script is `eslint src/lib/lesson src/routes/review` — the same two paths the flat config
scopes the rule to in `site/eslint.config.js`. The rest of the site is deliberately **not**
covered: the hedge rule is a pedagogy-surface rule, and widening it would turn marketing copy
into lint failures.

What the rule inspects, and what it deliberately does not:

| Node type | Covers |
|---|---|
| `Literal` | plain `'...'` / `"..."` strings in `.ts` / `.js` and in `<script>` blocks |
| `TemplateElement` | the static chunks of a `` `...` `` template literal |
| `SvelteText` | raw text between tags in a `.svelte` template |
| `SvelteLiteral` | Svelte attribute-value text |

**Comments are exempt.** JS/TS line and block comments and Svelte `<!-- ... -->` comments are
range-checked out, which is why the "this ward would have saved you" example in the
`epistemic.ts` header doc-comment does not trip the rule. This is also why the guide's old
`rg` recipe needed a "hits in `epistemic.ts` are fine" caveat and the lint rule does not.

Two escape hatches, both in the rule's options schema:

- `allow: ['...']` — literal substrings that suppress the rule for the chunk containing them.
  For prose that legitimately quotes a banned phrase outside a comment.
- `disablePhrases: ['you-will']` — turn off one row of the table by its id.

Prefer neither. The right fix for a real hit is (a) rewording or (b) routing the claim through
`hedge()`. The rule is defense-in-depth *behind* the helper, not a substitute for it — it can
only catch phrases someone wrote by hand, so a claim that skips `hedge()` while avoiding every
banned phrase still gets through. `hedge()` remains the load-bearing control.

The old grep still works as a quick manual spot-check:

```
rg -n "you will|would have|should always|correct answer" site/src/routes/review/preview site/src/lib/lesson
```

_Rule added 2026-08-17 (htw-9ip)._
