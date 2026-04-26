## M4 Retrieval Check — Design

**Status:** Design approved, ready for implementation plan.
**Date:** 2026-04-17
**Predecessor:** Plan B RSSM v1 (`docs/plan_b_eval_report_2026-04-17.md`).
**Successor:** M5 First Teaching Surface.

## Purpose

Build a kNN retrieval layer over the Plan B latent state and prove that nearest-neighbour cohorts of held-out mid-game anchors carry **informative outcome splits**. The headline necessary gate is **mean cohort outcome entropy ≥ 0.7 bits** at headline `k=64`, restricted to mid-game anchors (minutes 10–25), measured on both the game-cold and player-cold holdouts.

This is the gate before M5 (teaching surfaces). A high-entropy cohort is what a critical-point lesson lives inside: half the cohort won, half lost, from the same state — meaning a *decision* differentiated them. If the latent doesn't produce such cohorts, M5's lessons inherit broken signal.

## Non-goals

Out of scope for this design:

- The lesson generator (M5).
- Critical-point detection or anchor scoring.
- Production-grade kNN serving (FAISS, sharded indices, low-latency lookup) — see `howtowin.lol-l2z`.
- Re-training Plan B. The trained checkpoint `data/model_checkpoints/plan_b_full_best.pt` is consumed as-is.
- Any user-facing surface.

## Alignment note

This doc inherits the stricter upstream goal that retrieval should test whether the latent captures **in-game state**, not just draft or identity proxies. That means the entropy threshold is **necessary but not sufficient**:

- entropy that is too low means the cohort is one-sided and unteachable;
- entropy that is too high (e.g. near random-k ≈ 1.0) means the cohort is too loose to be useful;
- therefore the headline threshold must be interpreted alongside the baseline comparisons below.

## Architecture

### Vector key

For each anchor `t` of each game `g`, the retrieval key is the concatenation:

```
key(g, t) = [ h_t  ‖  μ_q(z_t) ]      ∈ ℝ^544
```

Where:
- `h_t` ∈ ℝ^512 — deterministic GRU state from the RSSM core.
- `μ_q(z_t)` ∈ ℝ^32 — **mean** of the posterior `q(z_t | h_t, o_t)`, not a sample. Removes per-call stochasticity from retrieval.

Rationale: the four decoder heads consume `[h_t ‖ z_t]`, so anchors that are L2-close in this space are anchors the heads treat similarly. Two anchors with the same head-input vector → same predicted distributions → genuinely "similar in the model's eyes."

### Whitening

Per-dimension z-score normalisation, parameters fit on the **training-corpus subset of keys** (not the holdouts):

```
μ_dim ∈ ℝ^544,  σ_dim ∈ ℝ^544    (computed once over corpus rows)
key_white = (key - μ_dim) / σ_dim
```

Distance is plain L2 on `key_white` (equivalent to diagonal Mahalanobis on the raw key). Whitening corrects for two asymmetries: (i) `h_t` is 512-dim with no magnitude prior so it would otherwise dominate the 32-dim `μ_q(z_t)` block by dimensionality alone; (ii) the GRU has no normalisation prior so its raw scale is whatever training produced. The whitening parameters travel inside the saved index artifact so any downstream consumer (M5) inherits the same calibration.

### Corpus and leak discipline

Index every anchor of every **training-split** game (not just mid-game). Restricting the corpus to minutes 10–25 was the alternative; we chose all-anchors so the eval also tests whether the latent space correctly clusters by game state vs. by time-since-start. If the model learned good representations, mid-game queries will retrieve mid-game cohorts naturally; if it didn't, that's a finding.

**Hard exclusions** from the index (mirrors Plan B training/eval splits):
- All match IDs in the game-cold holdout (`data/splits/plan_a_holdout.txt`).
- All match IDs of any puuid in the player-cold holdout (`data/splits/plan_b_cold_holdout.txt`).

Resulting corpus size: ~12.8k games × ~35 anchors ≈ 450k vectors. Single tensor: 450k × 544 × 4B ≈ 980 MB float32. Holds in RAM and on a 4 GB GPU.

### kNN backend

PyTorch native, exact, batched on GPU:

```python
# query_white: (Bq, 544),  corpus_white: (Bc, 544)
dists = torch.cdist(query_white, corpus_white)   # (Bq, Bc)
_, idx = dists.topk(k, dim=1, largest=False)     # (Bq, k)
```

Query batches sized so `(Bq × Bc × 4B)` fits in VRAM; for 4 GB and `Bc=450k`, `Bq≈2000` is safe (~3.6 GB). Held-out queries are ≪10k total per holdout, so a single pass per holdout is feasible. Estimated wall time on a GTX 1650 Super: < 2 minutes per full eval pass.

Future swap to FAISS is trivial since the artifact is just (corpus_tensor, whitening_params, row_metadata) — see `howtowin.lol-l2z`.

### Index artifact

A single `torch.save`'d dict written to `data/retrieval/plan_b_index.pt` (gitignored — large; HF-uploaded per the `huggingface.md` rules):

```python
{
    "corpus_white": Tensor (N_anchors, 544),     # whitened keys
    "mu_dim": Tensor (544,),                     # whitening mean
    "sigma_dim": Tensor (544,),                  # whitening std
    "row_match_id": list[str] (N_anchors,),      # per-row match_id
    "row_anchor_minute": Tensor (N_anchors,),    # per-row minute index
    "row_blue_win": Tensor (N_anchors,) int8,    # per-row source-game blue-team outcome
    "checkpoint_sha": str,                       # plan_b checkpoint hash
    "code_sha": str,                             # git HEAD when built
    "built_at": int,                             # unix ts
}
```

`row_team_outcome` is one-hot for the **game-level** outcome of the source game (since cohort entropy is over game outcomes, not per-anchor outcomes). The encoding is two-column to make it trivial to choose blue-win vs red-win when the query has a polarity.

## Eval

### Queries

For each holdout split (`game_cold` and `player_cold`), build query keys for **mid-game anchors only** (minutes 10–25 inclusive). Same encoder pass as the corpus, same whitening params, no exclusions (the holdout games are the queries, not the corpus).

### Metric

For each query anchor `q`, retrieve top-k corpus rows. Compute the **cohort win-rate** as

```
p = fraction of cohort rows whose source-game has blue_win == 1
H(p) = -p log2(p) - (1-p) log2(1-p)
```

with `H(0) = H(1) = 0`. The metric does not reference the query's own outcome — entropy is a property of the cohort alone. Because `H(p) = H(1-p)`, this is identical to a "cohort-query agreement entropy" and the choice of blue-as-reference team is arbitrary. Headline number: **mean H(p) across all mid-game query anchors of the holdout.**

### Headline gate

Necessary condition: `mean_H ≥ 0.7 bits` at `k=64`, **on both** game-cold and player-cold holdouts. (≥0.7 bits ⇒ cohort win-rate roughly in [0.20, 0.80].)

Interpretation then uses the baseline contrast below: the model should sit well **below** random-k entropy, and its position relative to static-only / frame-features tells us whether the learned latent adds value beyond trivial proxies.

### k sweep

Run the eval at `k ∈ {16, 32, 64, 128, 256}` and report the entropy curve. Guards against the headline depending on a flattering `k`. Strictly cheap: unrestricted eval performs one max-`k` retrieval pass per index and slices the wider top-k prefixes; skill-aware retrieval remains separately evaluated unless equivalence is proven.

### Runtime knobs

`python -m model.cli retrieval-eval` chooses `cuda` when CUDA is visible and otherwise uses `cpu`. Use `--device cpu` for CPU-only smoke runs or to avoid GPU scheduling, and use `--query-batch-size N` to control the exact `cdist + topk` query batch size. GPU acceleration helps the exact distance search only after repeated k-sweep work has been removed; it is not a replacement for single wide retrieval plus slicing. Tests and acceptance checks must not require a CUDA device.

### Per-minute breakdown

Report mean H(p) per minute bucket (10..25) at headline `k=64`. We hypothesize entropy will peak somewhere in the middle of the window (early states are uncommitted; very late states are decided), but this is a **diagnostic expectation**, not a standalone pass/fail gate. A flat or inverted curve is a finding worth investigating.

### Baselines

Run on the same query set, same metric, same `k`-sweep:

1. **Random-k** — sample `k` corpus rows uniformly without replacement. Negative control: expected ~1.0 bit. Confirms that 0.7-ish entropy is *not* trivial — the model must produce cohorts in [0.6, 0.9], i.e., tighter than random.
2. **Static-only** — kNN on the static-stream encoder output (10× champ embed + patch vector encoder), whitened the same way. Tests that dynamic state matters beyond draft.
3. **Frame-features** — kNN on a hand-crafted per-anchor numeric vector: `[gold, total_gold, xp, level, cs, jungle_cs, kills, deaths, assists]` for each of 10 participants = 90-dim, plus `minute` = 91-dim, whitened. Tests that the learned latent beats trivial state-matching.

Reported as a single entropy-vs-k curve plot (4 lines: model + 3 baselines; the random-k baseline is itself the uniform-1.0 reference) plus the per-minute table at headline `k=64` for the model only.

### Sanity test (not a baseline)

Self-retrieval: pick any 100 training-corpus anchors at random, query them against the same index, assert top-1 returned row index equals the query row index for every one. Verifies index correctness end-to-end.

### Deferred baselines (filed as beads)

- `howtowin.lol-rzz` — player-only baseline (player-stream encoder only).
- `howtowin.lol-cow` — `h_t`-only and `μ_q(z_t)`-only ablation baselines.

Both are diagnostics; neither is needed to defend the headline gate.

## Reporting

A markdown report committed to `docs/m4_retrieval_eval_report_<YYYY-MM-DD>.md`:

- Headline gate result: pass/fail, mean entropy at `k=64` for both holdouts.
- k-sweep table (model + baselines).
- Per-minute entropy table (model only, k=64).
- Index artifact metadata (checkpoint sha, code sha, corpus size, build time).
- Any deviations from this spec, called out explicitly.

The HF upload (`hf upload <user>/<repo> data/retrieval/plan_b_index.pt`) is the artifact handoff to M5.

## Files

**New:**
- `code/model/retrieval.py` — `build_index(model, train_match_ids, exclude_match_ids) → IndexBundle`, `query(index, queries, k) → (cohort_idx, cohort_dists)`, `save(index, path)`, `load(path) → IndexBundle`. Includes whitening fit/apply.
- `code/model/baselines/static_only_index.py` — `build_static_only_index(model, train_match_ids, exclude_match_ids)` reusing the static-stream encoder from PlanBModel.
- `code/model/baselines/frame_features_index.py` — `build_frame_features_index(train_match_ids, exclude_match_ids)` reading per-anchor numeric stats directly from the SQLite frames table; no model dependency.
- `code/model/m4_eval.py` — eval harness: builds queries, runs k-sweep, computes entropy, prints tables, writes report. Calls each baseline.
- `code/tests/model/test_retrieval.py` — unit tests for whitening (mean=0/std=1 on the corpus), self-retrieval sanity, leak-exclusion.
- `code/tests/model/test_baselines_static_only.py` — shape + leak tests.
- `code/tests/model/test_baselines_frame_features.py` — shape + minute-restriction tests.
- `code/tests/model/test_m4_eval.py` — entropy computation tests on a tiny synthetic corpus with known cohorts.

**Modified:**
- `code/model/cli.py` — add `retrieval-build` and `retrieval-eval` subcommands.
- `.gitignore` — add `data/retrieval/` if not already covered.

## Open questions

None. All design decisions agreed during brainstorming:
- Vector key: `[h_t ‖ μ_q(z_t)]` (option C in brainstorm).
- Corpus: all training anchors, no minute restriction (option A).
- k: sweep `{16, 32, 64, 128, 256}`, headline 64 (option D).
- Backend: PyTorch native exact (option A), FAISS deferred to bead.
- Distance: per-dim whitened L2 (option B).
- Baselines: random-k, static-only, frame-features (with player-only and ht/zt ablations deferred to beads).

## Not revisiting

Locked unless new evidence arrives:

- Game-level outcome as the cohort label (not anchor-level — the sequence model emits per-anchor outcomes but the *teachable* signal is the game's eventual result).
- Mid-game-only query restriction (10–25). Early-game and late-game anchors are either trivially split or trivially decided per the parent spec; including them would dilute the headline.
- Both holdouts must clear the gate independently. A model that retrieves well only on game-cold (seen players, new games) but not on player-cold (new players entirely) hasn't learned generalising state representations.
