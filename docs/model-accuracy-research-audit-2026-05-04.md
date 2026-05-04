# Model Accuracy Research Audit

Date: 2026-05-04

Scope: prime the local papers in `research/`, compare their assumptions against the current howtowin.lol model/retrieval strategy, and identify categorical flaws that may interfere with model accuracy or lesson quality.

## Research Corpus

| File | Main relevance |
|---|---|
| `research/188226.pdf` | Continuous League outcome prediction with RNNs. Accuracy rises with game time, splits must be by whole match, and patch/meta drift is a major ceiling. |
| `research/2001.11274v2.pdf` | Player skill, champion proficiency, recent history, momentum, feature scaling, and post-target Elo measurement can materially affect prediction. |
| `research/2006.15521v1.pdf` | League win probabilities need calibration, not just accuracy. Input-dependent uncertainty makes standard confidence scaling insufficient. |
| `research/2108.02799v1.pdf` | Player-champion mastery and champion-specific win rate are strong pre-game predictors even after matchmaking. |
| `research/Smart_kills_and_worthless_deaths_eSports_analytics.pdf` | Action value should be based on win-probability change, not raw event counts. Correlated frame samples require split/sampling discipline. |
| `research/tjal24a_public.pdf` | Item/action evaluation should use mean initial win probability and mean win probability added to correct selection bias. |

## Current Strategy

howtowin.lol is pursuing a three-stream RSSM world model plus retrieval-backed lesson generation.

| Layer | Current implementation |
|---|---|
| Data/model | Static context, player features, and dynamic event/frame streams in `code/model/dataset.py`, `code/model/encoders.py`, and `code/model/plan_b_model.py`. |
| Objective | RSSM trained with next-event, outcome, next-decision, next-frame, KL, and rollout auxiliary losses in `code/model/plan_b_train.py`. |
| Evaluation | Outcome AUC by minute, game-cold/player-cold splits, and frozen-minute-0 leak probe in `code/model/plan_b_eval.py`. |
| Retrieval | kNN over `[h_t || post_mu_t]`, using cohort outcome entropy as a retrieval gate in `code/model/retrieval.py` and `code/model/m4_eval.py`. |
| Lesson surface | Highest-entropy mid-game anchors are surfaced in `code/model/lesson.py`; the differentiating action is not decoded yet. |

The overall direction is sound: the repo already recognizes leakage, player-cold validation, uncertainty, skill confounding, and retrieval entropy. The biggest remaining risk is that several of these are documented but not fully enforced in the current training/evaluation/lesson path.

## Priority Findings

### 1. Calibration is missing from the current evaluation contract

Evidence:
- `code/model/plan_b_eval.py` reports AUC by minute but does not compute ECE or reliability diagrams.
- `docs/plan_b_eval_report_2026-04-17.md` reports AUC and leak probes, not probability calibration.

Why it matters:
- The calibration paper shows League probabilities contain input-dependent uncertainty. AUC can improve while probability estimates remain wrong.
- howtowin.lol's downstream claims depend on probabilities, not just ranking: win-probability added, cohort confidence, and item/action value all require calibrated probabilities.

Recommended check:
- Add ECE and reliability diagrams by minute, rank band, queue, and side.
- Treat calibration failure as blocking for win-probability-added or lesson-confidence claims.

### 2. Retrieval is not skill-aware yet

Evidence:
- `code/model/retrieval.py` stores only `match_id`, `anchor_minute`, and `blue_win` per row.
- `code/model/lesson.py` queries cohorts without rank, role, champion, or patch filters.
- `docs/superpowers/plans/2026-04-24-skill-causal-team-map.md` already identifies retrieval confounding as the main downstream bottleneck.

Why it matters:
- A high-entropy cohort can mix very different skill bands. The resulting lesson may be true for Challenger-like players but not actionable for the target player.
- The research brief explicitly flags skill-stratified retrieval as required for educational systems.

Recommended check:
- Add rank-band metadata to retrieval rows.
- Query same-band first, widen to adjacent bands when effective `k` is too small, then fall back with rank-distance penalties.
- Report unrestricted entropy and skill-aware entropy side by side.

### 3. Rank and LP may be post-hoc features

Evidence:
- `code/model/player_features.py` reads `players.rank_tier` and `players.lp` directly.
- The crawler/parser stores current player rank information, but the audit did not find evidence that rank is snapshot as of the target match's creation time.

Why it matters:
- White and Romano identify this exact issue: Elo/rank fetched after a target match can leak or blur target-time skill.
- If rank is current rather than match-time, it can encode future results and produce misleading player priors.

Recommended check:
- Prefer match-time rank snapshots when available.
- If unavailable, mark current rank as a noisy post-hoc proxy and measure rank-only AUC to estimate leakage strength.

### 4. Training player aggregates can include the target match or future matches

Evidence:
- `player_feature_vector()` applies the causal cutoff only when an excluded match belongs to the puuid.
- `plan_b_train_loop()` passes the union of validation and cold matches as `exclude_match_ids`, not each training match itself.

Why it matters:
- A training sample can include its own outcome and minute-10 stats in its player aggregate.
- Evaluation is cleaner because holdouts are excluded, so training and evaluation feature distributions can diverge.

Recommended check:
- For every sample, compute player features with a cutoff at that match's `created_at`, or materialize player features as-of match time.
- Add a train-time self-leak probe: train with player features only and compare train-vs-holdout AUC.

### 5. Current rollout evaluation is not pure prior-only imagination

Evidence:
- `code/model/plan_b_eval.py` calls `rollout_prior_single()` with real `event_window_positions`, offsets, and counts.
- `code/model/plan_b_model.py` advances rollout state through the actual future event-window embeddings before sampling prior states.

Why it matters:
- Reported rollout top-5 can be inflated by future event information.
- This weakens any claim that the model can imagine counterfactual futures, which is the main reason RSSM was chosen over a simpler transformer.

Recommended check:
- Add a strict prior-only rollout mode that does not consume real future event windows.
- Report teacher-forced rollout and pure prior rollout separately.

### 6. Entropy is necessary but not sufficient for lesson retrieval

Evidence:
- `docs/m4_retrieval_eval_report_2026-04-17.md` shows the model clears the entropy gate but trails the frame-feature baseline at every reported `k`.
- The same report shows random retrieval has even higher entropy, proving entropy alone does not validate state similarity.

Why it matters:
- High entropy can mean an ambiguous or noisy cohort, not a near-identical state with meaningful decision divergence.
- The lesson surface currently selects high-entropy anchors without proving the cohort is similar for the right reasons.

Recommended check:
- Pair entropy with neighborhood quality diagnostics: frame-feature distance, rank-band effective `k`, champion/role overlap, and outcome calibration.
- Add a retrieval gate that requires both entropy and similarity quality.

### 7. Lesson generation does not yet correct action-selection bias

Evidence:
- `code/model/lesson.py` chooses highest-entropy anchors but does not compute mean initial win probability or mean win probability added for candidate actions.

Why it matters:
- The Jalovaara thesis shows raw winrate is biased because actions/items are chosen in non-random contexts.
- Current lessons can identify a split state, but not yet justify that a specific decision caused the split.

Recommended check:
- For each surfaced decision family, report sample size, mean initial win probability, mean win probability added, and rank-stratified stability.
- Use causal/action validation as a filter before user-facing recommendations.

### 8. Static patch context is still scaffolding

Evidence:
- `code/model/static_features.py` includes picks, side, queue, region, time bucket, and patch vector.
- `code/model/patch_params.py` defaults to scaffolding mode where rich champion/item regions are zeroed or limited.

Why it matters:
- Item recommendations and cross-patch generalization require real item/champion parameter conditioning.
- Patch/meta drift is repeatedly identified in the papers as a major source of prediction error.

Recommended check:
- Add patch-heldout or future-patch evaluation once corpus size supports it.
- Expand rich numeric patch features before trusting item recommendations across patches.

### 9. Player-champion proficiency is missing from implemented player features

Evidence:
- `PLAYER_FEATURE_DIM` currently covers rank, LP, corpus winrate, KDA, CS@10, gold@10, wards, and role mix.
- It does not include champion mastery, player-champion winrate, games on champion, or recent games on champion.

Why it matters:
- Do et al. found player-champion experience can predict pre-game outcomes at roughly 75 percent accuracy in their setting.
- Ignoring player-champion proficiency leaves a large skill signal out of both prediction and retrieval.

Recommended check:
- Add player-champion features as-of match time: mastery, games on champion, champion-specific winrate, and recent champion usage.
- Run ablations: no player features, rank only, champion proficiency only, and all player features.

### 10. Existing splits do not test temporal or patch generalization

Evidence:
- `code/model/cold_holdout.py` uses SHA1 sampling for game-cold and player-cold splits.

Why it matters:
- Random holdouts test unseen matches and unseen players, but not future-meta accuracy.
- A model can pass random holdouts while failing after a patch or meta shift.

Recommended check:
- Add chronological split evaluation.
- Add patch-heldout evaluation when the corpus spans enough patches.

## Suggested Evaluation Matrix

| Risk | Minimum diagnostic |
|---|---|
| Miscalibration | ECE/reliability by minute, rank band, queue, side. |
| Skill confounding | Skill-aware retrieval entropy and effective `k`; rank-only and rank-swapped probes. |
| Player-feature leakage | Player-only AUC, frozen-dynamic probe, as-of-match feature materialization. |
| Weak rollout validity | Separate teacher-forced, event-conditioned, and strict prior-only rollout metrics. |
| Selection bias | Mean initial win probability and mean win probability added per action family. |
| Patch/meta drift | Chronological and patch-heldout evaluation. |
| Missing champion proficiency | Ablations for rank, champion proficiency, recent-history, and full player feature set. |

## Bottom Line

The model strategy is plausible, but the current system is still at risk of optimizing predictive shortcuts rather than robust, actionable teaching signal. The most urgent blockers are calibration, skill-aware retrieval, target-time player features, strict prior-only rollout evaluation, and action-level selection-bias correction.

Until those are measured, treat AUC and entropy passes as proof of signal, not proof of accurate counterfactual lessons.
