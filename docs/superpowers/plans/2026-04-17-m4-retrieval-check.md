# M4 Retrieval Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a kNN retrieval layer over Plan B's `[h_t ‖ μ_q(z_t)]` latent and prove that mean cohort outcome entropy ≥ 0.7 bits at `k=64` on both holdouts (game-cold, player-cold), restricted to mid-game anchors (minutes 10–25).

**Architecture:** Reuse Plan B's trained checkpoint and `MatchDataset` unchanged. Add a `retrieval.py` module that runs the model over training games, extracts `[h_t ‖ μ_q(z_t)]` per anchor, fits per-dim whitening, and saves a single PyTorch tensor index. The eval (`m4_eval.py`) runs PyTorch-native batched `cdist + topk` queries from each holdout game's mid-game anchors against the index, computes binary cohort entropy, and reports a k-sweep plus per-minute table. Three contrast baselines (random-k, static-only encoder, raw frame features) anchor what 0.7 bits *means*. Two ablations (player-only, h-only/z-only) are deferred to beads.

**Tech Stack:** Python 3.12, `uv`, PyTorch 2.11.0+cu130, numpy, pytest, SQLite (existing). No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md`. Keep it open while implementing.

---

## File Structure

**New files:**
- `code/model/retrieval.py` — `Whitener`, `IndexBundle` dataclass, `encode_game_keys`, `build_index`, `save_index`, `load_index`, `query_index`.
- `code/model/baselines/__init__.py`
- `code/model/baselines/static_only_index.py` — `build_static_only_index` reusing the `StaticContextEncoder` + `static_to_h` from PlanBModel.
- `code/model/baselines/frame_features_index.py` — `build_frame_features_index` reading per-anchor frame stats directly (no model dependency).
- `code/model/m4_eval.py` — `binary_entropy`, `cohort_entropies`, `mean_entropy_at_k`, `per_minute_entropy_table`, `run_m4_eval`.
- `code/tests/model/test_retrieval.py`
- `code/tests/model/test_baselines_static_only.py`
- `code/tests/model/test_baselines_frame_features.py`
- `code/tests/model/test_m4_eval.py`

**Modified:**
- `code/model/plan_b_model.py` — add per-anchor `h` to forward output dict (additive).
- `code/model/cli.py` — add `retrieval-build` and `retrieval-eval` subcommands.
- `code/tests/model/test_plan_b_model.py` — assert new `h` key shape.
- `.gitignore` — add `data/retrieval/` if not already covered by `data/*` rules.

**Key constants (top of `retrieval.py`):**
- `D_H = 512` (carried from `plan_b_model.py`)
- `D_Z = 32` (carried)
- `KEY_DIM = D_H + D_Z`  # 544
- `INDEX_DIR = os.path.join(<...repo...>/data/retrieval)`
- `DEFAULT_INDEX_PATH = os.path.join(INDEX_DIR, "plan_b_index.pt")`
- `MID_GAME_MINUTES = range(10, 26)`  # 10..25 inclusive
- `K_SWEEP = (16, 32, 64, 128, 256)`
- `HEADLINE_K = 64`
- `HEADLINE_GATE_BITS = 0.7`

---

## Task 0: Environment sanity

**Files:**
- None (pre-flight).

- [ ] **Step 1: Verify env and tests pass**

Run:
```bash
cd code && uv sync && uv run pytest tests/ -q
```
Expected: all tests pass (Plan B baseline). If any fail, fix before proceeding.

- [ ] **Step 2: Verify plan-b checkpoint is reachable**

Run:
```bash
ls -la ../data/model_checkpoints/plan_b_full_best.pt 2>&1 || echo "MISSING"
```
If `MISSING`: pull from HF / GPU host before continuing. The eval depends on this checkpoint. (Plan: `hf download <user>/<repo> plan_b_full_best.pt --local-dir ../data/model_checkpoints/`. Confirm exact HF path with the user before downloading.)

- [ ] **Step 3: Confirm holdout split files exist**

Run:
```bash
ls -la ../data/splits/plan_a_holdout.txt ../data/splits/plan_b_cold_holdout.txt
```
Expected: both files exist. If `plan_b_cold_holdout.txt` is missing: `cd code && uv run python -m model.cli cold-build`.

---

## Task 1: Expose per-anchor `h_t` from PlanBModel

**Files:**
- Modify: `code/model/plan_b_model.py`
- Modify: `code/tests/model/test_plan_b_model.py`

The current `PlanBModel.forward` returns only `h_final`. We need `h_t` for every anchor `t` to build the retrieval key `[h_t ‖ μ_q(z_t)]`. The change is purely additive: collect `h` per step into a list and stack into the output dict under key `"h"`.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_plan_b_model.py`:
```python
def test_forward_returns_per_anchor_h(fixture_match_id):
    from model.plan_b_model import PlanBModel, D_H
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)
    T = out["n_anchors"]
    assert "h" in out, "PlanBModel.forward must return per-anchor h"
    assert out["h"].shape == (1, T, D_H), \
        f"expected (1, {T}, {D_H}), got {tuple(out['h'].shape)}"
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_plan_b_model.py::test_forward_returns_per_anchor_h -v`
Expected: FAIL with `KeyError: 'h'` or `assert "h" in out`.

- [ ] **Step 3: Modify `PlanBModel.forward`**

In `code/model/plan_b_model.py`, find the per-anchor RSSM loop. Add an `h_all` list alongside `z_all`, append `h.clone()` *after* the `self.rssm.step(...)` call (so the recorded `h` is the value used to compute the prior/posterior at this step), stack at the end, and add to the return dict.

Locate the loop:
```python
        h = self.static_to_h(static)  # (B, D_H)
        z = torch.zeros(B, D_Z, device=h.device)

        post_mu_all, post_logvar_all = [], []
        prior_mu_all, prior_logvar_all = [], []
        z_all = []

        for t in range(T):
            h = self.rssm.step(h, z, action_summary[:, t])
            pr_mu, pr_lv = self.rssm.prior(h)
            po_mu, po_lv = self.rssm.posterior(h, obs[:, t])
            z = reparameterize(po_mu, po_lv)
            post_mu_all.append(po_mu)
            post_logvar_all.append(po_lv)
            prior_mu_all.append(pr_mu)
            prior_logvar_all.append(pr_lv)
            z_all.append(z)
```

Change to:
```python
        h = self.static_to_h(static)  # (B, D_H)
        z = torch.zeros(B, D_Z, device=h.device)

        post_mu_all, post_logvar_all = [], []
        prior_mu_all, prior_logvar_all = [], []
        z_all, h_all = [], []

        for t in range(T):
            h = self.rssm.step(h, z, action_summary[:, t])
            h_all.append(h)
            pr_mu, pr_lv = self.rssm.prior(h)
            po_mu, po_lv = self.rssm.posterior(h, obs[:, t])
            z = reparameterize(po_mu, po_lv)
            post_mu_all.append(po_mu)
            post_logvar_all.append(po_lv)
            prior_mu_all.append(pr_mu)
            prior_logvar_all.append(pr_lv)
            z_all.append(z)
```

Then in the post-loop stacking block, add:
```python
        H = torch.stack(h_all, dim=1)                 # (B, T, D_H)
```

And add `"h": H,` to the return dict alongside `"z": Z,`.

- [ ] **Step 4: Run test, verify pass**

Run: `cd code && uv run pytest tests/model/test_plan_b_model.py -v`
Expected: all tests pass, including the new one.

- [ ] **Step 5: Run the full plan_b test set to confirm no regression**

Run: `cd code && uv run pytest tests/model/test_plan_b_model.py tests/model/test_plan_b_eval.py tests/model/test_plan_b_train.py -q`
Expected: all pass. The change is additive so existing consumers (eval, train) ignore the new key.

- [ ] **Step 6: Commit**

```bash
git add code/model/plan_b_model.py code/tests/model/test_plan_b_model.py
git commit -m "plan-b: expose per-anchor h_t in forward output for M4 retrieval"
```

---

## Task 2: Whitening primitive

**Files:**
- Create: `code/model/retrieval.py`
- Create: `code/tests/model/test_retrieval.py`

The whitening primitive fits per-dim mean/std on a corpus tensor and applies to query tensors. Plain z-score, no clipping, with `sigma` clamped from below to avoid div-by-zero on dead dims.

- [ ] **Step 1: Write the failing test**

Create `code/tests/model/test_retrieval.py`:
```python
import torch
import pytest


def test_whitener_fit_makes_corpus_unit_normal():
    from model.retrieval import Whitener
    torch.manual_seed(0)
    corpus = torch.randn(1024, 32) * 7.5 + 3.0
    w = Whitener.fit(corpus)
    out = w.apply(corpus)
    assert torch.allclose(out.mean(dim=0), torch.zeros(32), atol=1e-5)
    assert torch.allclose(out.std(dim=0, unbiased=False),
                          torch.ones(32), atol=1e-3)


def test_whitener_apply_uses_fitted_params_not_query_stats():
    from model.retrieval import Whitener
    torch.manual_seed(0)
    corpus = torch.randn(1024, 8) * 2.0 + 5.0
    w = Whitener.fit(corpus)
    query = torch.full((4, 8), 5.0)  # exactly the corpus mean
    out = w.apply(query)
    assert torch.allclose(out, torch.zeros(4, 8), atol=1e-5)


def test_whitener_clamps_zero_std_dims():
    from model.retrieval import Whitener
    corpus = torch.zeros(100, 4)
    corpus[:, 0] = torch.arange(100, dtype=torch.float32)  # nonzero std
    # Dims 1, 2, 3 are constant → std==0; whitener must not produce NaN/Inf.
    w = Whitener.fit(corpus)
    out = w.apply(corpus)
    assert torch.isfinite(out).all()


def test_whitener_roundtrips_through_state_dict():
    from model.retrieval import Whitener
    torch.manual_seed(0)
    corpus = torch.randn(64, 16)
    w = Whitener.fit(corpus)
    sd = w.state_dict()
    w2 = Whitener.from_state_dict(sd)
    assert torch.equal(w.mu, w2.mu)
    assert torch.equal(w.sigma, w2.sigma)
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.retrieval'`.

- [ ] **Step 3: Create `code/model/retrieval.py` with the `Whitener` class**

```python
"""kNN retrieval over Plan B latent state.

Builds an index of [h_t || mu_q(z_t)] per training anchor, fits per-dim
whitening, and exposes batched cdist+topk queries. Backed by PyTorch
tensors only — no FAISS dependency. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import os
from dataclasses import dataclass
import torch

D_H = 512
D_Z = 32
KEY_DIM = D_H + D_Z  # 544

INDEX_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "retrieval"
)
DEFAULT_INDEX_PATH = os.path.join(INDEX_DIR, "plan_b_index.pt")

MID_GAME_MINUTES = tuple(range(10, 26))  # 10..25 inclusive
K_SWEEP = (16, 32, 64, 128, 256)
HEADLINE_K = 64
HEADLINE_GATE_BITS = 0.7

_SIGMA_FLOOR = 1e-6


class Whitener:
    """Per-dim z-score normalization. Params fit once on the corpus."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def fit(cls, corpus: torch.Tensor) -> "Whitener":
        mu = corpus.mean(dim=0)
        sigma = corpus.std(dim=0, unbiased=False).clamp(min=_SIGMA_FLOOR)
        return cls(mu=mu, sigma=sigma)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma

    def state_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "Whitener":
        return cls(mu=sd["mu"], sigma=sd["sigma"])
```

- [ ] **Step 4: Run test, verify pass**

Run: `cd code && uv run pytest tests/model/test_retrieval.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/retrieval.py code/tests/model/test_retrieval.py
git commit -m "m4: whitening primitive for retrieval index"
```

---

## Task 3: Binary entropy

**Files:**
- Create: `code/model/m4_eval.py`
- Create: `code/tests/model/test_m4_eval.py`

The eval-side primitive. Binary entropy in bits, vectorised, with `H(0) = H(1) = 0` and numerical safety.

- [ ] **Step 1: Write the failing test**

Create `code/tests/model/test_m4_eval.py`:
```python
import math
import torch


def test_binary_entropy_at_half_is_one_bit():
    from model.m4_eval import binary_entropy
    p = torch.tensor([0.5])
    h = binary_entropy(p)
    assert torch.allclose(h, torch.tensor([1.0]), atol=1e-6)


def test_binary_entropy_at_endpoints_is_zero():
    from model.m4_eval import binary_entropy
    p = torch.tensor([0.0, 1.0])
    h = binary_entropy(p)
    assert torch.allclose(h, torch.tensor([0.0, 0.0]), atol=1e-6)


def test_binary_entropy_is_symmetric_around_half():
    from model.m4_eval import binary_entropy
    a = binary_entropy(torch.tensor([0.2]))
    b = binary_entropy(torch.tensor([0.8]))
    assert torch.allclose(a, b, atol=1e-6)


def test_binary_entropy_at_seventy_thirty_matches_formula():
    from model.m4_eval import binary_entropy
    p = 0.7
    expected = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    h = binary_entropy(torch.tensor([p]))
    assert abs(h.item() - expected) < 1e-6
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.m4_eval'`.

- [ ] **Step 3: Create `code/model/m4_eval.py`**

```python
"""M4 retrieval-check eval harness.

Builds queries from holdout splits, runs k-sweep entropy, prints
per-minute table, runs baseline indexes, writes report. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import math
import torch

_LOG2 = math.log(2.0)
_EPS = 1e-12


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-element binary entropy in bits. Returns 0 at p in {0, 1}."""
    p = p.clamp(min=_EPS, max=1.0 - _EPS)
    h_nats = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return h_nats / _LOG2
```

- [ ] **Step 4: Run test, verify pass**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/m4_eval.py code/tests/model/test_m4_eval.py
git commit -m "m4: binary entropy primitive"
```

---

## Task 4: `encode_game_keys` — extract per-anchor latent for one game

**Files:**
- Modify: `code/model/retrieval.py`
- Modify: `code/tests/model/test_retrieval.py`

Given a `PlanBModel` and a single-game batch, return per-anchor `(keys, anchor_minutes, blue_win)`. Decoupled from index-build so it's testable on the fixture match.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_retrieval.py`:
```python
def test_encode_game_keys_shapes(fixture_match_id):
    from model.retrieval import encode_game_keys, KEY_DIM
    from model.plan_b_model import PlanBModel
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    model = PlanBModel(max_puuids=len(idx) + 1)
    keys, minutes, blue_win = encode_game_keys(model, batch)
    T = batch["anchor_positions"].size(1)
    assert keys.shape == (T, KEY_DIM)
    assert minutes.shape == (T,)
    assert minutes.dtype == torch.int64
    assert blue_win.shape == ()         # scalar tensor
    assert blue_win.dtype == torch.int8
    assert int(blue_win.item()) in (0, 1)


def test_encode_game_keys_minutes_are_monotonic(fixture_match_id):
    from model.retrieval import encode_game_keys
    from model.plan_b_model import PlanBModel
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    model = PlanBModel(max_puuids=len(idx) + 1)
    _, minutes, _ = encode_game_keys(model, batch)
    diffs = minutes[1:] - minutes[:-1]
    assert (diffs >= 0).all(), \
        "anchor minutes must be non-decreasing"
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_retrieval.py::test_encode_game_keys_shapes -v`
Expected: FAIL with `ImportError: cannot import name 'encode_game_keys'`.

- [ ] **Step 3: Add `encode_game_keys` to `code/model/retrieval.py`**

```python
@torch.no_grad()
def encode_game_keys(model, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model.forward on a single-game batch (B=1) and return:

    keys:       (T, KEY_DIM) — concat of [h_t, post_mu_t] per anchor.
    minutes:    (T,) int64 — anchor minute = round(token_timestamp / 60000).
    blue_win:   () int8 — game-level outcome (1 if blue won, else 0).
    """
    assert batch["tokens"].size(0) == 1, "encode_game_keys expects B=1"
    was_training = model.training
    model.eval()
    out = model(batch)
    model.train(was_training)

    T = out["n_anchors"]
    h = out["h"][0]                # (T, D_H)
    post_mu = out["post_mu"][0]    # (T, D_Z)
    keys = torch.cat([h, post_mu], dim=-1)  # (T, KEY_DIM)

    anchor_pos = batch["anchor_positions"][0]   # (T,)
    ts = batch["token_timestamps"][0]           # (L,)
    anchor_ts = ts.gather(0, anchor_pos.long()) # (T,) ms
    minutes = (anchor_ts / 60000.0).round().to(torch.int64)

    blue_win = torch.tensor(int(batch["outcome"][0].item()), dtype=torch.int8)
    return keys, minutes, blue_win
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_retrieval.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/retrieval.py code/tests/model/test_retrieval.py
git commit -m "m4: encode_game_keys — per-anchor latent extraction"
```

---

## Task 5: `IndexBundle` dataclass + save/load

**Files:**
- Modify: `code/model/retrieval.py`
- Modify: `code/tests/model/test_retrieval.py`

The artifact format. A frozen dataclass holding the corpus tensor, whitener, per-row metadata, and provenance fields.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_retrieval.py`:
```python
def test_index_bundle_roundtrips(tmp_path):
    from model.retrieval import IndexBundle, Whitener, KEY_DIM, save_index, load_index
    torch.manual_seed(0)
    N = 32
    corpus = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.arange(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="deadbeef",
        code_sha="cafef00d",
        built_at=1700000000,
    )
    p = tmp_path / "idx.pt"
    save_index(bundle, str(p))
    loaded = load_index(str(p))
    assert torch.equal(bundle.corpus_white, loaded.corpus_white)
    assert torch.equal(bundle.whitener.mu, loaded.whitener.mu)
    assert torch.equal(bundle.whitener.sigma, loaded.whitener.sigma)
    assert bundle.row_match_id == loaded.row_match_id
    assert torch.equal(bundle.row_anchor_minute, loaded.row_anchor_minute)
    assert torch.equal(bundle.row_blue_win, loaded.row_blue_win)
    assert loaded.checkpoint_sha == "deadbeef"
    assert loaded.code_sha == "cafef00d"
    assert loaded.built_at == 1700000000
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_retrieval.py::test_index_bundle_roundtrips -v`
Expected: FAIL with `ImportError: cannot import name 'IndexBundle'`.

- [ ] **Step 3: Add `IndexBundle` + save/load to `code/model/retrieval.py`**

```python
@dataclass
class IndexBundle:
    corpus_white: torch.Tensor             # (N, KEY_DIM) float32
    whitener: Whitener
    row_match_id: list[str]                # per-row source match_id
    row_anchor_minute: torch.Tensor        # (N,) int64
    row_blue_win: torch.Tensor             # (N,) int8 in {0, 1}
    checkpoint_sha: str
    code_sha: str
    built_at: int                          # unix ts


def save_index(bundle: IndexBundle, path: str = DEFAULT_INDEX_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "corpus_white": bundle.corpus_white,
            "whitener": bundle.whitener.state_dict(),
            "row_match_id": bundle.row_match_id,
            "row_anchor_minute": bundle.row_anchor_minute,
            "row_blue_win": bundle.row_blue_win,
            "checkpoint_sha": bundle.checkpoint_sha,
            "code_sha": bundle.code_sha,
            "built_at": bundle.built_at,
        },
        path,
    )


def load_index(path: str = DEFAULT_INDEX_PATH) -> IndexBundle:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return IndexBundle(
        corpus_white=raw["corpus_white"],
        whitener=Whitener.from_state_dict(raw["whitener"]),
        row_match_id=list(raw["row_match_id"]),
        row_anchor_minute=raw["row_anchor_minute"],
        row_blue_win=raw["row_blue_win"],
        checkpoint_sha=raw["checkpoint_sha"],
        code_sha=raw["code_sha"],
        built_at=raw["built_at"],
    )
```

- [ ] **Step 4: Run test, verify pass**

Run: `cd code && uv run pytest tests/model/test_retrieval.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/retrieval.py code/tests/model/test_retrieval.py
git commit -m "m4: IndexBundle dataclass + save/load"
```

---

## Task 6: `build_index` — full corpus pass with leak exclusion

**Files:**
- Modify: `code/model/retrieval.py`
- Modify: `code/tests/model/test_retrieval.py`

Iterate over training match_ids (excluding both holdouts), encode each game's anchors, accumulate rows, fit the whitener on the raw corpus, whiten in place, and return an `IndexBundle`.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_retrieval.py`:
```python
def test_build_index_excludes_holdout_match_ids(fixture_match_id):
    from model.retrieval import build_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids={fixture_match_id},
        puuid_index=idx,
        device="cpu",
    )
    # The only candidate match was excluded — corpus should be empty.
    assert bundle.corpus_white.shape[0] == 0
    assert bundle.row_match_id == []


def test_build_index_corpus_is_whitened(fixture_match_id):
    from model.retrieval import build_index, KEY_DIM
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
    )
    N = bundle.corpus_white.shape[0]
    assert N > 0
    assert bundle.corpus_white.shape == (N, KEY_DIM)
    # With one game, whitening forces every row identical → mean ~ 0.
    assert torch.allclose(
        bundle.corpus_white.mean(dim=0), torch.zeros(KEY_DIM), atol=1e-4
    )


def test_build_index_records_per_row_metadata(fixture_match_id):
    from model.retrieval import build_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
    )
    N = bundle.corpus_white.shape[0]
    # All rows come from the one fixture match.
    assert all(mid == fixture_match_id for mid in bundle.row_match_id)
    assert len(bundle.row_match_id) == N
    assert bundle.row_anchor_minute.shape == (N,)
    assert bundle.row_blue_win.shape == (N,)
    # All rows share the same per-game outcome.
    assert int(bundle.row_blue_win.unique().numel()) == 1
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_retrieval.py::test_build_index_excludes_holdout_match_ids -v`
Expected: FAIL with `ImportError: cannot import name 'build_index'`.

- [ ] **Step 3: Add `build_index` to `code/model/retrieval.py`**

Add at the top of the file:
```python
import subprocess
import time

from torch.utils.data import DataLoader
from model.dataset import MatchDataset, collate_games
```

Then implement:
```python
def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def build_index(
    *,
    model,
    train_match_ids: list[str],
    exclude_match_ids: set[str],
    puuid_index: dict,
    device: str = "cpu",
    checkpoint_sha: str = "unknown",
    log_every: int = 100,
) -> IndexBundle:
    """Encode every training game's anchors, fit whitening, return bundle.

    train_match_ids: candidate corpus games.
    exclude_match_ids: subtracted from the corpus AND passed to MatchDataset
        for player-feature-leak discipline (matches Plan B training contract).
    """
    model.eval()
    model.to(device)

    eligible = [m for m in train_match_ids if m not in exclude_match_ids]

    rows_list: list[torch.Tensor] = []
    minutes_list: list[torch.Tensor] = []
    blue_win_list: list[int] = []
    match_id_list: list[str] = []

    if not eligible:
        empty = torch.zeros(0, KEY_DIM)
        zero_w = Whitener(mu=torch.zeros(KEY_DIM), sigma=torch.ones(KEY_DIM))
        return IndexBundle(
            corpus_white=empty,
            whitener=zero_w,
            row_match_id=[],
            row_anchor_minute=torch.zeros(0, dtype=torch.int64),
            row_blue_win=torch.zeros(0, dtype=torch.int8),
            checkpoint_sha=checkpoint_sha,
            code_sha=_git_head_sha(),
            built_at=int(time.time()),
        )

    ds = MatchDataset(
        eligible, puuid_index=puuid_index,
        exclude_match_ids=exclude_match_ids,
        cache_size=1,  # we only touch each game once
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)

    for i, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            keys, minutes, blue_win = encode_game_keys(model, batch)
        except Exception as e:
            # Skip pathological games rather than aborting the whole build.
            print(f"[build_index] skipping match {eligible[i]}: {e}")
            continue
        T = keys.shape[0]
        rows_list.append(keys.cpu())
        minutes_list.append(minutes.cpu())
        blue_win_list.extend([int(blue_win.item())] * T)
        match_id_list.extend([eligible[i]] * T)
        if (i + 1) % log_every == 0:
            print(f"[build_index] {i + 1}/{len(eligible)} games encoded")

    corpus_raw = torch.cat(rows_list, dim=0) if rows_list else torch.zeros(0, KEY_DIM)
    minutes_all = torch.cat(minutes_list, dim=0) if minutes_list else torch.zeros(0, dtype=torch.int64)
    blue_win_all = torch.tensor(blue_win_list, dtype=torch.int8)

    whitener = Whitener.fit(corpus_raw) if corpus_raw.shape[0] > 0 \
               else Whitener(mu=torch.zeros(KEY_DIM), sigma=torch.ones(KEY_DIM))
    corpus_white = whitener.apply(corpus_raw)

    return IndexBundle(
        corpus_white=corpus_white,
        whitener=whitener,
        row_match_id=match_id_list,
        row_anchor_minute=minutes_all,
        row_blue_win=blue_win_all,
        checkpoint_sha=checkpoint_sha,
        code_sha=_git_head_sha(),
        built_at=int(time.time()),
    )
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_retrieval.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/retrieval.py code/tests/model/test_retrieval.py
git commit -m "m4: build_index — full corpus pass with leak exclusion"
```

---

## Task 7: `query_index` — batched cdist+topk

**Files:**
- Modify: `code/model/retrieval.py`
- Modify: `code/tests/model/test_retrieval.py`

Take a query tensor `(Q, KEY_DIM)` of *raw* (un-whitened) keys and return per-query top-k cohort indices and distances, batching to fit the GPU.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_retrieval.py`:
```python
def test_query_index_returns_self_at_rank_one_for_corpus_rows():
    from model.retrieval import (
        IndexBundle, Whitener, KEY_DIM, query_index,
    )
    torch.manual_seed(0)
    N = 100
    corpus_raw = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus_raw)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus_raw),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.arange(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
    )
    queries_raw = corpus_raw[:5]  # exact match against corpus rows 0..4
    cohort_idx, _dists = query_index(bundle, queries_raw, k=3,
                                     device="cpu", batch_size=16)
    assert cohort_idx.shape == (5, 3)
    # Top-1 for each query must be the matching row.
    assert torch.equal(cohort_idx[:, 0], torch.arange(5, dtype=cohort_idx.dtype))


def test_query_index_returns_distinct_cohorts():
    from model.retrieval import (
        IndexBundle, Whitener, KEY_DIM, query_index,
    )
    torch.manual_seed(1)
    N = 64
    corpus_raw = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus_raw)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus_raw),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.zeros(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
    )
    q = torch.randn(8, KEY_DIM)
    cohort_idx, dists = query_index(bundle, q, k=4, device="cpu", batch_size=4)
    assert cohort_idx.shape == (8, 4)
    assert dists.shape == (8, 4)
    # Distances must be monotonically non-decreasing across the k axis.
    diffs = dists[:, 1:] - dists[:, :-1]
    assert (diffs >= -1e-5).all()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_retrieval.py::test_query_index_returns_self_at_rank_one_for_corpus_rows -v`
Expected: FAIL with `ImportError: cannot import name 'query_index'`.

- [ ] **Step 3: Add `query_index` to `code/model/retrieval.py`**

```python
@torch.no_grad()
def query_index(
    bundle: IndexBundle,
    queries_raw: torch.Tensor,
    *,
    k: int,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cohort_idx, dists) of shape (Q, k) each.

    queries_raw: (Q, KEY_DIM) — raw keys, whitened internally with the
        bundle's fitted Whitener.
    """
    Q = queries_raw.shape[0]
    if Q == 0:
        return (
            torch.zeros(0, k, dtype=torch.long),
            torch.zeros(0, k, dtype=torch.float32),
        )

    corpus = bundle.corpus_white.to(device)
    mu = bundle.whitener.mu.to(device)
    sigma = bundle.whitener.sigma.to(device)

    out_idx = torch.empty(Q, k, dtype=torch.long)
    out_d = torch.empty(Q, k, dtype=torch.float32)

    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        qb = queries_raw[start:end].to(device)
        qb_white = (qb - mu) / sigma
        d = torch.cdist(qb_white, corpus)        # (b, N)
        d_top, idx_top = d.topk(k, dim=1, largest=False)
        out_idx[start:end] = idx_top.cpu()
        out_d[start:end] = d_top.cpu()

    return out_idx, out_d
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_retrieval.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/retrieval.py code/tests/model/test_retrieval.py
git commit -m "m4: query_index — batched cdist+topk against whitened corpus"
```

---

## Task 8: `cohort_entropies` and `mean_entropy_at_k` — per-query and aggregate

**Files:**
- Modify: `code/model/m4_eval.py`
- Modify: `code/tests/model/test_m4_eval.py`

`cohort_entropies` takes the top-k indices and the corpus's `blue_win` labels and returns one entropy per query. `mean_entropy_at_k` aggregates.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_m4_eval.py`:
```python
def test_cohort_entropies_pure_cohort_has_zero_entropy():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3]])  # (1 query, k=4)
    blue_win = torch.tensor([1, 1, 1, 1], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert h.shape == (1,)
    assert abs(h[0].item() - 0.0) < 1e-6


def test_cohort_entropies_balanced_cohort_has_one_bit():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3]])
    blue_win = torch.tensor([1, 1, 0, 0], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert abs(h[0].item() - 1.0) < 1e-6


def test_cohort_entropies_per_query_independent():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    blue_win = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert h.shape == (2,)
    assert abs(h[0].item() - 0.0) < 1e-6  # all wins
    assert abs(h[1].item() - 1.0) < 1e-6  # half-half


def test_mean_entropy_at_k_averages():
    from model.m4_eval import mean_entropy_at_k
    cohort_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    blue_win = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int8)
    mean_h = mean_entropy_at_k(cohort_idx, blue_win)
    assert abs(mean_h - 0.5) < 1e-6
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py -v`
Expected: 4 passed (existing) + 4 failed with `ImportError: cannot import name 'cohort_entropies'`.

- [ ] **Step 3: Add `cohort_entropies` and `mean_entropy_at_k` to `code/model/m4_eval.py`**

```python
def cohort_entropies(
    cohort_idx: torch.Tensor, blue_win: torch.Tensor
) -> torch.Tensor:
    """Per-query cohort outcome entropy in bits.

    cohort_idx: (Q, k) long — top-k corpus row indices per query.
    blue_win: (N,) int8 — per-corpus-row source-game outcome.
    Returns: (Q,) float entropies.
    """
    cohort_labels = blue_win[cohort_idx].float()    # (Q, k)
    p = cohort_labels.mean(dim=1)
    return binary_entropy(p)


def mean_entropy_at_k(
    cohort_idx: torch.Tensor, blue_win: torch.Tensor
) -> float:
    h = cohort_entropies(cohort_idx, blue_win)
    if h.numel() == 0:
        return float("nan")
    return float(h.mean().item())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/m4_eval.py code/tests/model/test_m4_eval.py
git commit -m "m4: cohort_entropies and mean_entropy_at_k"
```

---

## Task 9: Static-only baseline index

**Files:**
- Create: `code/model/baselines/__init__.py`
- Create: `code/model/baselines/static_only_index.py`
- Create: `code/tests/model/test_baselines_static_only.py`

Reuse the `StaticContextEncoder` + `static_to_h` projection from PlanBModel. The "key" for a game is the same value the RSSM uses to seed `h_0` — a single 512-dim vector per *game* (not per anchor). Each anchor of the game shares the same key. This tests "draft-only retrieval."

- [ ] **Step 1: Write the failing test**

Create `code/tests/model/test_baselines_static_only.py`:
```python
import torch


def test_build_static_only_index_returns_one_row_per_anchor(fixture_match_id):
    from model.baselines.static_only_index import build_static_only_index
    from model.plan_b_model import PlanBModel, D_H
    from model.dataset import build_puuid_index, MatchDataset, collate_games
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_static_only_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
    )
    # Same anchor count as the model's index would produce.
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    T = batch["anchor_positions"].size(1)
    assert bundle.corpus_white.shape == (T, D_H)
    # All T rows for one game must be identical (static is per-game, not per-anchor).
    first = bundle.corpus_white[0]
    for r in range(1, T):
        assert torch.allclose(bundle.corpus_white[r], first, atol=1e-6)


def test_build_static_only_index_excludes(fixture_match_id):
    from model.baselines.static_only_index import build_static_only_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_static_only_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids={fixture_match_id},
        puuid_index=idx,
        device="cpu",
    )
    assert bundle.corpus_white.shape[0] == 0
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_baselines_static_only.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.baselines'`.

- [ ] **Step 3: Create `code/model/baselines/__init__.py`**

```python
```

(Empty file.)

- [ ] **Step 4: Create `code/model/baselines/static_only_index.py`**

```python
"""Static-only retrieval baseline: kNN on PlanBModel.static_to_h(static_enc(static)).

The 'key' is a per-GAME 512-dim vector (the value the RSSM uses to seed
h_0), copied per anchor for shape parity with the headline index.

See spec: docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import time
import torch
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games
from model.retrieval import (
    IndexBundle, Whitener, _git_head_sha,
)
from model.plan_b_model import D_H


@torch.no_grad()
def _encode_game_static_key(model, batch) -> torch.Tensor:
    """Return (D_H,) static-only key for a single-game batch."""
    static_enc = model.static_enc(batch["static"])         # (1, D_MODEL)
    h0 = model.static_to_h(static_enc)                     # (1, D_H)
    return h0[0].detach().cpu()


@torch.no_grad()
def build_static_only_index(
    *,
    model,
    train_match_ids: list[str],
    exclude_match_ids: set[str],
    puuid_index: dict,
    device: str = "cpu",
    log_every: int = 100,
) -> IndexBundle:
    model.eval()
    model.to(device)

    eligible = [m for m in train_match_ids if m not in exclude_match_ids]
    if not eligible:
        empty = torch.zeros(0, D_H)
        zero_w = Whitener(mu=torch.zeros(D_H), sigma=torch.ones(D_H))
        return IndexBundle(
            corpus_white=empty, whitener=zero_w,
            row_match_id=[],
            row_anchor_minute=torch.zeros(0, dtype=torch.int64),
            row_blue_win=torch.zeros(0, dtype=torch.int8),
            checkpoint_sha="static_only", code_sha=_git_head_sha(),
            built_at=int(time.time()),
        )

    ds = MatchDataset(eligible, puuid_index=puuid_index,
                      exclude_match_ids=exclude_match_ids, cache_size=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)

    rows: list[torch.Tensor] = []
    minutes_list: list[torch.Tensor] = []
    blue_win_list: list[int] = []
    mid_list: list[str] = []

    for i, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            key = _encode_game_static_key(model, batch)
        except Exception as e:
            print(f"[static_only] skipping {eligible[i]}: {e}")
            continue
        T = batch["anchor_positions"].size(1)

        anchor_pos = batch["anchor_positions"][0].long()
        ts = batch["token_timestamps"][0]
        anchor_ts = ts.gather(0, anchor_pos)
        minutes = (anchor_ts / 60000.0).round().to(torch.int64).cpu()
        outcome = int(batch["outcome"][0].item())

        rows.append(key.unsqueeze(0).expand(T, -1).clone())
        minutes_list.append(minutes)
        blue_win_list.extend([outcome] * T)
        mid_list.extend([eligible[i]] * T)

        if (i + 1) % log_every == 0:
            print(f"[static_only] {i + 1}/{len(eligible)} games encoded")

    corpus_raw = torch.cat(rows, dim=0)
    minutes_all = torch.cat(minutes_list, dim=0)
    blue_win_all = torch.tensor(blue_win_list, dtype=torch.int8)

    whitener = Whitener.fit(corpus_raw)
    corpus_white = whitener.apply(corpus_raw)

    return IndexBundle(
        corpus_white=corpus_white,
        whitener=whitener,
        row_match_id=mid_list,
        row_anchor_minute=minutes_all,
        row_blue_win=blue_win_all,
        checkpoint_sha="static_only",
        code_sha=_git_head_sha(),
        built_at=int(time.time()),
    )
```

Then add a sibling helper to `code/model/baselines/static_only_index.py` for query keys:

```python
@torch.no_grad()
def encode_static_only_query_key(model, batch) -> torch.Tensor:
    """(T, D_H) — copy the per-game static key per anchor."""
    key = _encode_game_static_key(model, batch)            # (D_H,)
    T = batch["anchor_positions"].size(1)
    return key.unsqueeze(0).expand(T, -1).clone()
```

- [ ] **Step 5: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_baselines_static_only.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add code/model/baselines/__init__.py code/model/baselines/static_only_index.py code/tests/model/test_baselines_static_only.py
git commit -m "m4: static-only baseline index"
```

---

## Task 10: Frame-features baseline index

**Files:**
- Create: `code/model/baselines/frame_features_index.py`
- Create: `code/tests/model/test_baselines_frame_features.py`

A hand-crafted per-anchor 91-dim vector: `[gold, total_gold, xp, level, cs, jungle_cs, kills, deaths, assists]` per participant (90 dims) + minute (1 dim). Doesn't depend on the model — reads from SQLite.

- [ ] **Step 1: Write the failing test**

Create `code/tests/model/test_baselines_frame_features.py`:
```python
import torch


FRAME_BASELINE_DIM = 91  # 9 stats × 10 participants + 1 minute


def test_frame_features_index_dim_is_91(fixture_match_id):
    from model.baselines.frame_features_index import build_frame_features_index
    bundle = build_frame_features_index(
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
    )
    assert bundle.corpus_white.shape[1] == FRAME_BASELINE_DIM


def test_frame_features_index_excludes(fixture_match_id):
    from model.baselines.frame_features_index import build_frame_features_index
    bundle = build_frame_features_index(
        train_match_ids=[fixture_match_id],
        exclude_match_ids={fixture_match_id},
    )
    assert bundle.corpus_white.shape[0] == 0


def test_frame_features_index_records_anchor_minutes(fixture_match_id):
    from model.baselines.frame_features_index import build_frame_features_index
    bundle = build_frame_features_index(
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
    )
    N = bundle.corpus_white.shape[0]
    assert bundle.row_anchor_minute.shape == (N,)
    assert bundle.row_blue_win.shape == (N,)
    # All from one match.
    assert all(mid == fixture_match_id for mid in bundle.row_match_id)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_baselines_frame_features.py -v`
Expected: FAIL with `ModuleNotFoundError: ...frame_features_index`.

- [ ] **Step 3: Create `code/model/baselines/frame_features_index.py`**

```python
"""Frame-features retrieval baseline: kNN on a per-anchor hand-crafted
numeric vector built directly from the SQLite frames table.

Vector layout per anchor (91 dims):
  [gold, total_gold, xp, level, cs, jungle_cs, kills, deaths, assists]
    × 10 participants  =  90 dims
  + minute_index                     1 dim

See spec: docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import time
import numpy as np
import torch

from db import get_conn
from model.retrieval import IndexBundle, Whitener, _git_head_sha

FRAME_BASELINE_STATS = (
    "current_gold", "total_gold", "xp", "level", "cs", "jungle_cs",
    "kills", "deaths", "assists",
)
N_STATS = len(FRAME_BASELINE_STATS)
N_PARTICIPANTS = 10
FRAME_BASELINE_DIM = N_STATS * N_PARTICIPANTS + 1  # 91


def _read_game_frames(match_id: str):
    conn = get_conn()
    try:
        rows = conn.execute(
            f"SELECT timestamp_ms, participant_slot, "
            f"{', '.join(FRAME_BASELINE_STATS)} "
            f"FROM frames WHERE match_id = ? "
            f"AND participant_slot BETWEEN 1 AND 10 "
            f"ORDER BY timestamp_ms ASC",
            (match_id,),
        ).fetchall()
        game = conn.execute(
            "SELECT winning_team FROM games WHERE match_id = ?", (match_id,)
        ).fetchone()
    finally:
        conn.close()
    return rows, game


def _build_per_anchor_vectors(rows):
    """Pivot frame rows to (T_anchors, 91)."""
    by_ts: dict[int, dict[int, dict]] = {}
    for r in rows:
        by_ts.setdefault(r["timestamp_ms"], {})[r["participant_slot"]] = r
    timestamps = sorted(by_ts)
    T = len(timestamps)
    out = np.zeros((T, FRAME_BASELINE_DIM), dtype=np.float32)
    for ti, ts in enumerate(timestamps):
        for slot in range(1, N_PARTICIPANTS + 1):
            r = by_ts[ts].get(slot)
            if r is None:
                continue
            base = (slot - 1) * N_STATS
            for si, stat in enumerate(FRAME_BASELINE_STATS):
                v = r[stat]
                out[ti, base + si] = float(v) if v is not None else 0.0
        out[ti, -1] = float(ts) / 60000.0  # minute index
    minutes = np.array([round(ts / 60000.0) for ts in timestamps], dtype=np.int64)
    return out, minutes


def build_frame_features_index(
    *,
    train_match_ids: list[str],
    exclude_match_ids: set[str],
    log_every: int = 100,
) -> IndexBundle:
    eligible = [m for m in train_match_ids if m not in exclude_match_ids]
    if not eligible:
        empty = torch.zeros(0, FRAME_BASELINE_DIM)
        zero_w = Whitener(
            mu=torch.zeros(FRAME_BASELINE_DIM),
            sigma=torch.ones(FRAME_BASELINE_DIM),
        )
        return IndexBundle(
            corpus_white=empty, whitener=zero_w,
            row_match_id=[],
            row_anchor_minute=torch.zeros(0, dtype=torch.int64),
            row_blue_win=torch.zeros(0, dtype=torch.int8),
            checkpoint_sha="frame_features",
            code_sha=_git_head_sha(),
            built_at=int(time.time()),
        )

    rows: list[np.ndarray] = []
    minutes_list: list[np.ndarray] = []
    blue_win_list: list[int] = []
    mid_list: list[str] = []

    for i, mid in enumerate(eligible):
        try:
            frame_rows, game = _read_game_frames(mid)
            if not frame_rows or game is None:
                continue
            vecs, minutes = _build_per_anchor_vectors(frame_rows)
            outcome = 1 if game["winning_team"] == 100 else 0
            T = vecs.shape[0]
            rows.append(vecs)
            minutes_list.append(minutes)
            blue_win_list.extend([outcome] * T)
            mid_list.extend([mid] * T)
        except Exception as e:
            print(f"[frame_features] skipping {mid}: {e}")
            continue
        if (i + 1) % log_every == 0:
            print(f"[frame_features] {i + 1}/{len(eligible)} games encoded")

    corpus_raw = torch.from_numpy(np.concatenate(rows, axis=0)) if rows \
                 else torch.zeros(0, FRAME_BASELINE_DIM)
    minutes_all = torch.from_numpy(np.concatenate(minutes_list, axis=0)) if minutes_list \
                  else torch.zeros(0, dtype=torch.int64)
    blue_win_all = torch.tensor(blue_win_list, dtype=torch.int8)

    whitener = Whitener.fit(corpus_raw)
    corpus_white = whitener.apply(corpus_raw)

    return IndexBundle(
        corpus_white=corpus_white,
        whitener=whitener,
        row_match_id=mid_list,
        row_anchor_minute=minutes_all,
        row_blue_win=blue_win_all,
        checkpoint_sha="frame_features",
        code_sha=_git_head_sha(),
        built_at=int(time.time()),
    )


def encode_frame_features_query(match_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (T, 91), (T,) keys + minutes for a single game's anchors."""
    frame_rows, _ = _read_game_frames(match_id)
    if not frame_rows:
        return torch.zeros(0, FRAME_BASELINE_DIM), torch.zeros(0, dtype=torch.int64)
    vecs, minutes = _build_per_anchor_vectors(frame_rows)
    return torch.from_numpy(vecs), torch.from_numpy(minutes)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_baselines_frame_features.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/baselines/frame_features_index.py code/tests/model/test_baselines_frame_features.py
git commit -m "m4: frame-features baseline index"
```

---

## Task 11: Random-k cohort sampler + per-minute table

**Files:**
- Modify: `code/model/m4_eval.py`
- Modify: `code/tests/model/test_m4_eval.py`

`random_k_cohort_indices` returns Q×k random indices into the corpus. `per_minute_entropy_table` groups query entropies by their *query anchor's* minute.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_m4_eval.py`:
```python
def test_random_k_cohort_indices_shape_and_range():
    from model.m4_eval import random_k_cohort_indices
    torch.manual_seed(0)
    idx = random_k_cohort_indices(Q=10, k=4, N=100, seed=0)
    assert idx.shape == (10, 4)
    assert (idx >= 0).all() and (idx < 100).all()


def test_random_k_cohort_indices_deterministic_under_seed():
    from model.m4_eval import random_k_cohort_indices
    a = random_k_cohort_indices(Q=10, k=4, N=100, seed=42)
    b = random_k_cohort_indices(Q=10, k=4, N=100, seed=42)
    assert torch.equal(a, b)


def test_per_minute_entropy_table_groups_by_query_minute():
    from model.m4_eval import per_minute_entropy_table
    cohort_h = torch.tensor([0.5, 0.7, 0.9, 1.0])
    query_minutes = torch.tensor([10, 10, 15, 20])
    table = per_minute_entropy_table(cohort_h, query_minutes,
                                     minutes=(10, 15, 20))
    assert table[10] == (0.5 + 0.7) / 2
    assert table[15] == 0.9
    assert table[20] == 1.0
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py -v`
Expected: 8 passed (existing) + 3 failed with import errors.

- [ ] **Step 3: Add the new functions to `code/model/m4_eval.py`**

```python
def random_k_cohort_indices(*, Q: int, k: int, N: int, seed: int) -> torch.Tensor:
    """(Q, k) of corpus indices sampled uniformly without per-row replacement."""
    g = torch.Generator()
    g.manual_seed(seed)
    out = torch.empty(Q, k, dtype=torch.long)
    for q in range(Q):
        # without-replacement within a single cohort; with-replacement across queries.
        out[q] = torch.randperm(N, generator=g)[:k]
    return out


def per_minute_entropy_table(
    cohort_h: torch.Tensor,
    query_minutes: torch.Tensor,
    minutes,
) -> dict[int, float]:
    """Mean cohort entropy bucketed by query anchor minute."""
    out = {}
    for m in minutes:
        mask = (query_minutes == m)
        if mask.any():
            out[int(m)] = float(cohort_h[mask].mean().item())
        else:
            out[int(m)] = float("nan")
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/m4_eval.py code/tests/model/test_m4_eval.py
git commit -m "m4: random-k cohort sampler + per-minute entropy table"
```

---

## Task 12: `run_m4_eval` — full eval orchestrator

**Files:**
- Modify: `code/model/m4_eval.py`
- Modify: `code/tests/model/test_m4_eval.py`

Glue: build queries from a holdout dataset, run the model index + baselines, return a structured result. Keeps CLI thin.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/model/test_m4_eval.py`:
```python
def test_run_m4_eval_returns_structured_result(fixture_match_id):
    from model.m4_eval import run_m4_eval
    from model.retrieval import build_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index

    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model, train_match_ids=[fixture_match_id],
        exclude_match_ids=set(), puuid_index=idx, device="cpu",
    )
    result = run_m4_eval(
        model=model, model_bundle=bundle,
        holdout_match_ids=[fixture_match_id],
        holdout_label="self_eval",
        puuid_index=idx,
        exclude_match_ids=set(),
        k_sweep=(2, 4),
        headline_k=4,
        device="cpu",
        run_baselines=False,
    )
    assert "model" in result
    assert "k_sweep" in result["model"]
    assert "per_minute_at_headline_k" in result["model"]
    for k in (2, 4):
        assert k in result["model"]["k_sweep"]
        v = result["model"]["k_sweep"][k]
        assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py::test_run_m4_eval_returns_structured_result -v`
Expected: FAIL with `ImportError: cannot import name 'run_m4_eval'`.

- [ ] **Step 3: Add `run_m4_eval` to `code/model/m4_eval.py`**

Add imports at the top of the file:
```python
import torch
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games
from model.retrieval import (
    encode_game_keys, query_index, MID_GAME_MINUTES,
)
```

Then implement:
```python
@torch.no_grad()
def _build_holdout_queries(model, holdout_match_ids, puuid_index,
                           exclude_match_ids, device, mid_minutes):
    """Encode every mid-game anchor of every holdout game.

    Returns (queries_raw, query_minutes) of shapes (Q, KEY_DIM), (Q,).
    """
    if not holdout_match_ids:
        from model.retrieval import KEY_DIM
        return (
            torch.zeros(0, KEY_DIM),
            torch.zeros(0, dtype=torch.int64),
        )

    ds = MatchDataset(holdout_match_ids, puuid_index=puuid_index,
                      exclude_match_ids=exclude_match_ids, cache_size=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)

    keys_list: list[torch.Tensor] = []
    mins_list: list[torch.Tensor] = []
    mid_minute_set = set(int(m) for m in mid_minutes)

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        keys, minutes, _ = encode_game_keys(model, batch)
        mask = torch.tensor([int(m.item()) in mid_minute_set for m in minutes],
                            dtype=torch.bool)
        if mask.any():
            keys_list.append(keys[mask].cpu())
            mins_list.append(minutes[mask].cpu())

    if not keys_list:
        from model.retrieval import KEY_DIM
        return (torch.zeros(0, KEY_DIM), torch.zeros(0, dtype=torch.int64))

    return torch.cat(keys_list, dim=0), torch.cat(mins_list, dim=0)


def _eval_one_index(bundle, queries_raw, query_minutes, *, k_sweep,
                    headline_k, headline_minutes, device, query_batch_size):
    sweep_means: dict[int, float] = {}
    headline_per_minute: dict[int, float] = {}
    headline_cohort_h = None
    for k in k_sweep:
        cohort_idx, _ = query_index(
            bundle, queries_raw, k=k, device=device,
            batch_size=query_batch_size,
        )
        sweep_means[int(k)] = mean_entropy_at_k(cohort_idx, bundle.row_blue_win)
        if int(k) == int(headline_k):
            cohort_h = cohort_entropies(cohort_idx, bundle.row_blue_win)
            headline_per_minute = per_minute_entropy_table(
                cohort_h, query_minutes, headline_minutes,
            )
            headline_cohort_h = cohort_h
    return {
        "k_sweep": sweep_means,
        "per_minute_at_headline_k": headline_per_minute,
        "headline_cohort_entropies": headline_cohort_h,
    }


@torch.no_grad()
def run_m4_eval(
    *,
    model,
    model_bundle,
    holdout_match_ids: list[str],
    holdout_label: str,
    puuid_index: dict,
    exclude_match_ids: set[str],
    k_sweep=(16, 32, 64, 128, 256),
    headline_k: int = 64,
    headline_minutes=tuple(range(10, 26)),
    device: str = "cpu",
    query_batch_size: int = 256,
    run_baselines: bool = True,
    static_only_bundle=None,
    frame_features_bundle=None,
    random_seed: int = 0,
) -> dict:
    queries_raw, query_minutes = _build_holdout_queries(
        model, holdout_match_ids, puuid_index, exclude_match_ids,
        device, headline_minutes,
    )
    out = {"holdout": holdout_label,
           "n_queries": int(queries_raw.shape[0])}

    out["model"] = _eval_one_index(
        model_bundle, queries_raw, query_minutes,
        k_sweep=k_sweep, headline_k=headline_k,
        headline_minutes=headline_minutes,
        device=device, query_batch_size=query_batch_size,
    )

    if run_baselines:
        # Random-k baseline (only sweep, no per-minute table — uniform by design).
        random_sweep: dict[int, float] = {}
        N = model_bundle.row_blue_win.shape[0]
        for k in k_sweep:
            cohort_idx = random_k_cohort_indices(
                Q=int(queries_raw.shape[0]), k=int(k), N=N,
                seed=random_seed + int(k),
            )
            random_sweep[int(k)] = mean_entropy_at_k(
                cohort_idx, model_bundle.row_blue_win,
            )
        out["random"] = {"k_sweep": random_sweep}

        if static_only_bundle is not None:
            from model.baselines.static_only_index import encode_static_only_query_key
            so_queries = _build_static_only_queries(
                model, holdout_match_ids, puuid_index, exclude_match_ids,
                device, headline_minutes,
            )
            out["static_only"] = _eval_one_index(
                static_only_bundle, so_queries[0], so_queries[1],
                k_sweep=k_sweep, headline_k=headline_k,
                headline_minutes=headline_minutes, device=device,
                query_batch_size=query_batch_size,
            )

        if frame_features_bundle is not None:
            ff_queries = _build_frame_features_queries(
                holdout_match_ids, headline_minutes,
            )
            out["frame_features"] = _eval_one_index(
                frame_features_bundle, ff_queries[0], ff_queries[1],
                k_sweep=k_sweep, headline_k=headline_k,
                headline_minutes=headline_minutes, device=device,
                query_batch_size=query_batch_size,
            )
    return out


@torch.no_grad()
def _build_static_only_queries(model, holdout_match_ids, puuid_index,
                               exclude_match_ids, device, mid_minutes):
    from model.baselines.static_only_index import encode_static_only_query_key
    from model.plan_b_model import D_H
    ds = MatchDataset(holdout_match_ids, puuid_index=puuid_index,
                      exclude_match_ids=exclude_match_ids, cache_size=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)
    keys_list, mins_list = [], []
    mid_set = set(int(m) for m in mid_minutes)
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        keys = encode_static_only_query_key(model, batch)  # (T, D_H)
        anchor_pos = batch["anchor_positions"][0].long()
        ts = batch["token_timestamps"][0]
        anchor_ts = ts.gather(0, anchor_pos)
        minutes = (anchor_ts / 60000.0).round().to(torch.int64).cpu()
        mask = torch.tensor([int(m.item()) in mid_set for m in minutes], dtype=torch.bool)
        if mask.any():
            keys_list.append(keys[mask].cpu())
            mins_list.append(minutes[mask])
    if not keys_list:
        return torch.zeros(0, D_H), torch.zeros(0, dtype=torch.int64)
    return torch.cat(keys_list, dim=0), torch.cat(mins_list, dim=0)


def _build_frame_features_queries(holdout_match_ids, mid_minutes):
    from model.baselines.frame_features_index import (
        encode_frame_features_query, FRAME_BASELINE_DIM,
    )
    keys_list, mins_list = [], []
    mid_set = set(int(m) for m in mid_minutes)
    for mid in holdout_match_ids:
        keys, minutes = encode_frame_features_query(mid)
        if keys.shape[0] == 0:
            continue
        mask = torch.tensor([int(m.item()) in mid_set for m in minutes], dtype=torch.bool)
        if mask.any():
            keys_list.append(keys[mask])
            mins_list.append(minutes[mask])
    if not keys_list:
        return torch.zeros(0, FRAME_BASELINE_DIM), torch.zeros(0, dtype=torch.int64)
    return torch.cat(keys_list, dim=0), torch.cat(mins_list, dim=0)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_m4_eval.py -v`
Expected: 12 passed.

- [ ] **Step 5: Run the full retrieval test set to confirm no regression**

Run: `cd code && uv run pytest tests/model/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add code/model/m4_eval.py code/tests/model/test_m4_eval.py
git commit -m "m4: run_m4_eval orchestrator (model + baselines)"
```

---

## Task 13: CLI subcommands `retrieval-build` and `retrieval-eval`

**Files:**
- Modify: `code/model/cli.py`

Two new subcommands:
- `retrieval-build` — load checkpoint, call `build_index`, save to `data/retrieval/plan_b_index.pt`. Optional `--baselines` flag also builds and saves the static-only and frame-features bundles.
- `retrieval-eval` — load index(es), run `run_m4_eval` on both holdouts, print tables and the final pass/fail line, write the markdown report.

- [ ] **Step 1: Add `cmd_retrieval_build` and `cmd_retrieval_eval` to `code/model/cli.py`**

Append the new functions before the `if __name__ == "__main__":` block:

```python
def _checkpoint_sha_short(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def cmd_retrieval_build(args):
    """Build the M4 retrieval index (model + optional baselines)."""
    import torch as _t
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    from model.retrieval import (
        build_index, save_index, DEFAULT_INDEX_PATH, INDEX_DIR,
    )

    ckpt_path = os.path.join(CHECKPOINT_DIR, "plan_b_full_best.pt")
    ckpt = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)
    sha = _checkpoint_sha_short(ckpt_path)

    train = load_split("train")
    val = load_split("holdout")
    cold = load_split("cold")
    exclude = set(val) | set(cold)
    puuid_index = build_puuid_index(train, max_puuids=max_puuids)

    device = "cuda" if _t.cuda.is_available() else "cpu"
    model = PlanBModel(max_puuids=max_puuids).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print(f"[retrieval-build] device={device} train={len(train)} "
          f"excluded={len(exclude)} ckpt_sha={sha}")
    bundle = build_index(
        model=model, train_match_ids=train,
        exclude_match_ids=exclude, puuid_index=puuid_index,
        device=device, checkpoint_sha=sha,
    )
    save_index(bundle, DEFAULT_INDEX_PATH)
    print(f"[retrieval-build] saved {DEFAULT_INDEX_PATH} "
          f"corpus_rows={bundle.corpus_white.shape[0]}")

    if args.baselines:
        from model.baselines.static_only_index import build_static_only_index
        from model.baselines.frame_features_index import build_frame_features_index

        so_path = os.path.join(INDEX_DIR, "plan_b_static_only_index.pt")
        ff_path = os.path.join(INDEX_DIR, "plan_b_frame_features_index.pt")

        print("[retrieval-build] static-only baseline...")
        so_bundle = build_static_only_index(
            model=model, train_match_ids=train,
            exclude_match_ids=exclude, puuid_index=puuid_index,
            device=device,
        )
        save_index(so_bundle, so_path)
        print(f"[retrieval-build] saved {so_path} "
              f"corpus_rows={so_bundle.corpus_white.shape[0]}")

        print("[retrieval-build] frame-features baseline...")
        ff_bundle = build_frame_features_index(
            train_match_ids=train, exclude_match_ids=exclude,
        )
        save_index(ff_bundle, ff_path)
        print(f"[retrieval-build] saved {ff_path} "
              f"corpus_rows={ff_bundle.corpus_white.shape[0]}")


def cmd_retrieval_eval(args):
    """Run the M4 eval and write a markdown report."""
    import json
    import time as _time
    import torch as _t
    from datetime import date

    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    from model.retrieval import (
        load_index, DEFAULT_INDEX_PATH, INDEX_DIR,
        K_SWEEP, HEADLINE_K, HEADLINE_GATE_BITS, MID_GAME_MINUTES,
    )
    from model.m4_eval import run_m4_eval

    ckpt_path = os.path.join(CHECKPOINT_DIR, "plan_b_full_best.pt")
    ckpt = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)

    train = load_split("train")
    val = load_split("holdout")
    cold = load_split("cold")
    exclude = set(val) | set(cold)
    puuid_index = build_puuid_index(train, max_puuids=max_puuids)

    device = "cuda" if _t.cuda.is_available() else "cpu"
    model = PlanBModel(max_puuids=max_puuids).to(device)
    model.load_state_dict(ckpt["state_dict"])

    model_bundle = load_index(DEFAULT_INDEX_PATH)
    so_bundle = ff_bundle = None
    if args.baselines:
        so_bundle = load_index(os.path.join(INDEX_DIR, "plan_b_static_only_index.pt"))
        ff_bundle = load_index(os.path.join(INDEX_DIR, "plan_b_frame_features_index.pt"))

    results = {}
    for label, ids in [("game_cold", val), ("player_cold", cold)]:
        print(f"\n=== {label.upper()} ({len(ids)} games) ===")
        r = run_m4_eval(
            model=model, model_bundle=model_bundle,
            holdout_match_ids=ids, holdout_label=label,
            puuid_index=puuid_index, exclude_match_ids=exclude,
            k_sweep=K_SWEEP, headline_k=HEADLINE_K,
            headline_minutes=MID_GAME_MINUTES,
            device=device, query_batch_size=128,
            run_baselines=args.baselines,
            static_only_bundle=so_bundle,
            frame_features_bundle=ff_bundle,
        )
        results[label] = r
        _print_eval_result(r)

    # Pass/fail.
    passed = (
        results["game_cold"]["model"]["k_sweep"][HEADLINE_K] >= HEADLINE_GATE_BITS
        and results["player_cold"]["model"]["k_sweep"][HEADLINE_K] >= HEADLINE_GATE_BITS
    )
    print(f"\nM4 GATE: {'PASS' if passed else 'FAIL'} "
          f"(headline_k={HEADLINE_K}, threshold={HEADLINE_GATE_BITS} bits)")

    # Write markdown report.
    report_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "docs",
    )
    report_path = os.path.join(
        report_dir, f"m4_retrieval_eval_report_{date.today().isoformat()}.md",
    )
    _write_report(report_path, results, passed)
    print(f"[retrieval-eval] wrote report to {report_path}")


def _print_eval_result(r: dict) -> None:
    n_q = r["n_queries"]
    print(f"  n_queries={n_q}")
    print("  Source            " + "  ".join(f"k={k:>3}" for k in r["model"]["k_sweep"]))
    for src in ("model", "static_only", "frame_features", "random"):
        if src in r:
            row = r[src]["k_sweep"]
            cells = "  ".join(f"{row[k]:.3f}" for k in row)
            print(f"  {src:<16}  {cells}")
    print("  Per-minute entropy at headline k (model only):")
    for m, v in sorted(r["model"]["per_minute_at_headline_k"].items()):
        print(f"    min {m:>2}: {v:.3f}" if v == v else f"    min {m:>2}:   nan")


def _write_report(path: str, results: dict, passed: bool) -> None:
    from datetime import date
    lines = [
        "# M4 Retrieval-Check Eval Report",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Gate:** mean cohort outcome entropy ≥ 0.7 bits at k=64 on both holdouts.",
        f"**Result:** **{'PASS' if passed else 'FAIL'}**",
        "",
    ]
    for label in ("game_cold", "player_cold"):
        r = results[label]
        lines.append(f"## {label} (n_queries={r['n_queries']})")
        ks = list(r["model"]["k_sweep"].keys())
        header = "| Source | " + " | ".join(f"k={k}" for k in ks) + " |"
        sep = "|" + "---|" * (len(ks) + 1)
        lines += ["", header, sep]
        for src in ("model", "static_only", "frame_features", "random"):
            if src in r:
                row = r[src]["k_sweep"]
                cells = " | ".join(f"{row[k]:.3f}" for k in ks)
                lines.append(f"| {src} | {cells} |")
        lines.append("")
        lines.append("### Per-minute entropy at headline k (model only)")
        lines.append("")
        lines.append("| Minute | Entropy (bits) |")
        lines.append("|---:|---:|")
        for m, v in sorted(r["model"]["per_minute_at_headline_k"].items()):
            cell = f"{v:.3f}" if v == v else "nan"
            lines.append(f"| {m} | {cell} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
```

- [ ] **Step 2: Wire up the argparse subparsers**

In the `if __name__ == "__main__":` block, add the subparsers below the existing ones (insert before `args = parser.parse_args()`):

```python
    p_rb = sub.add_parser("retrieval-build")
    p_rb.add_argument("--baselines", action="store_true",
                      help="also build static-only and frame-features indexes")

    p_re = sub.add_parser("retrieval-eval")
    p_re.add_argument("--baselines", action="store_true",
                      help="also evaluate baselines (requires --baselines on build)")
```

And add dispatch lines below the existing `elif args.cmd == ...` chain:

```python
    elif args.cmd == "retrieval-build":
        cmd_retrieval_build(args)
    elif args.cmd == "retrieval-eval":
        cmd_retrieval_eval(args)
```

- [ ] **Step 3: Verify CLI parses**

Run:
```bash
cd code && uv run python -m model.cli --help
uv run python -m model.cli retrieval-build --help
uv run python -m model.cli retrieval-eval --help
```
Expected: each prints help text without error.

- [ ] **Step 4: Commit**

```bash
git add code/model/cli.py
git commit -m "m4: CLI subcommands retrieval-build and retrieval-eval"
```

---

## Task 14: Local sanity run (CPU, tiny corpus)

**Files:**
- None (runtime validation).

A smoke test that exercises the full build → save → load → eval round-trip on a small corpus before consuming GPU time. Also serves as a documented runbook step for the GPU operator.

- [ ] **Step 1: Add `.gitignore` entry for the index directory if missing**

Run: `grep -F 'data/retrieval' /home/lunaris/build/howtowin.lol/.gitignore || echo 'data/retrieval/' >> /home/lunaris/build/howtowin.lol/.gitignore`

(If the line already exists, this is a no-op. If `data/*` is already in `.gitignore`, the new line is redundant but harmless.)

- [ ] **Step 2: Build a small index locally**

Run a one-shot Python script to limit the corpus to 50 games:

```bash
cd code && uv run python -c "
import os, torch
from model.plan_b_model import PlanBModel
from model.dataset import build_puuid_index, load_split
from model.retrieval import build_index, save_index, DEFAULT_INDEX_PATH

ckpt_path = '../data/model_checkpoints/plan_b_full_best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
max_puuids = ckpt.get('max_puuids', 20000)
train = load_split('train')[:50]
val = load_split('holdout')
cold = load_split('cold')
exclude = set(val) | set(cold)
idx = build_puuid_index(train, max_puuids=max_puuids)
model = PlanBModel(max_puuids=max_puuids)
model.load_state_dict(ckpt['state_dict'])
bundle = build_index(model=model, train_match_ids=train,
                     exclude_match_ids=exclude, puuid_index=idx,
                     device='cpu', checkpoint_sha='smoke')
save_index(bundle, DEFAULT_INDEX_PATH + '.smoke')
print('smoke corpus rows:', bundle.corpus_white.shape[0])
"
```
Expected: prints `smoke corpus rows: <several hundred>`.

- [ ] **Step 3: Run a tiny eval against the smoke index**

```bash
cd code && uv run python -c "
import torch
from datetime import date
from model.plan_b_model import PlanBModel
from model.dataset import build_puuid_index, load_split
from model.retrieval import load_index, DEFAULT_INDEX_PATH, MID_GAME_MINUTES
from model.m4_eval import run_m4_eval

ckpt_path = '../data/model_checkpoints/plan_b_full_best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
max_puuids = ckpt.get('max_puuids', 20000)
train = load_split('train')[:50]
val = load_split('holdout')[:4]
cold = load_split('cold')[:4]
exclude = set(load_split('holdout')) | set(load_split('cold'))
idx = build_puuid_index(train, max_puuids=max_puuids)
model = PlanBModel(max_puuids=max_puuids)
model.load_state_dict(ckpt['state_dict'])
bundle = load_index(DEFAULT_INDEX_PATH + '.smoke')
res = run_m4_eval(model=model, model_bundle=bundle,
                  holdout_match_ids=val, holdout_label='smoke_game_cold',
                  puuid_index=idx, exclude_match_ids=exclude,
                  k_sweep=(8, 16), headline_k=8,
                  headline_minutes=MID_GAME_MINUTES,
                  device='cpu', run_baselines=False)
print('smoke eval result:', res['n_queries'], 'queries')
print('k_sweep:', res['model']['k_sweep'])
"
```
Expected: prints `smoke eval result: <Q>` and a `k_sweep` dict with two finite values in [0.0, 1.0]. If entropy at small `k` and tiny corpus is suspiciously low or NaN, inspect before proceeding.

- [ ] **Step 4: Clean up smoke artifact**

Run: `rm -f ../data/retrieval/plan_b_index.pt.smoke`

- [ ] **Step 5: Commit (if .gitignore changed)**

```bash
git add .gitignore
git commit -m "m4: gitignore data/retrieval/"  || true
```

---

## Task 15: Full GPU eval + report

**Files:**
- Will be created/modified by the run: `data/retrieval/plan_b_index.pt`, `data/retrieval/plan_b_static_only_index.pt`, `data/retrieval/plan_b_frame_features_index.pt`, `docs/m4_retrieval_eval_report_<DATE>.md`.

- [ ] **Step 1: Push branch to origin**

```bash
git push
```

- [ ] **Step 2: Sync code + checkpoint to GPU host**

On the laptop:
```bash
scripts/sync-gpu.sh push
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol && git pull'
```

If the checkpoint isn't on the GPU host, also send it (or pull from HF on the GPU host directly):
```bash
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol && hf download <user>/<repo> plan_b_full_best.pt --local-dir data/model_checkpoints/'
```

- [ ] **Step 3: Build all three indexes on GPU**

```bash
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol/code && tmux new -d -s m4build "uv run python -m model.cli retrieval-build --baselines > /tmp/m4_build.log 2>&1"'
ssh "$HOWL_GPU_HOST" 'tail -f /tmp/m4_build.log'
```
Expected runtime: ~10–30 min on GTX 1650 Super for the model index (~12.8k games × ~35 anchors), faster for static-only (per-game), comparable for frame-features (DB-bound).

- [ ] **Step 4: Run the eval on GPU and capture output**

```bash
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol/code && uv run python -m model.cli retrieval-eval --baselines | tee /tmp/m4_eval.txt'
scp "$HOWL_GPU_HOST:/tmp/m4_eval.txt" .
```

- [ ] **Step 5: Pull the report and indexes back**

```bash
ssh "$HOWL_GPU_HOST" 'ls ~/build/howtowin.lol/docs/m4_retrieval_eval_report_*.md'
scp "$HOWL_GPU_HOST:~/build/howtowin.lol/docs/m4_retrieval_eval_report_*.md" docs/
scripts/sync-gpu.sh pull   # pulls anything else relevant
```

Optionally upload the indexes to HF (per `~/.claude/rules/huggingface.md`):
```bash
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol && hf upload <user>/<repo> data/retrieval/ --include "plan_b_*_index.pt"'
```

- [ ] **Step 6: Verify the gate**

Open the markdown report. Check:

1. **Headline gate:** `mean entropy ≥ 0.7 bits` at `k=64` on **both** `game_cold` and `player_cold` holdouts.
2. **Baseline contrast:** model k=64 entropy is **below** random-k entropy (which should be ≈1.0) — i.e., our cohorts are tighter than random.
3. **Baseline contrast:** model k=64 entropy is also strictly less than the static-only and frame-features baselines, *or* — if the model entropy is roughly equal to a baseline — the per-minute structure should still show meaningful variation. (A model that ties with frame-features hasn't learned beyond hand-crafted state.)
4. **Per-minute curve:** entropy is highest somewhere in the middle of the 10–25 window (mid-game is when outcomes are most uncertain). A flat or inverted curve is a finding worth opening as a follow-up.
5. **Sanity check the row counts:** model corpus rows ≈ static-only corpus rows ≈ frame-features corpus rows ± a few %. Large divergence suggests one of the encoders is dropping or duplicating games.

If any gate misses, file a follow-up bead with the specific number and decide with the user whether to iterate or accept.

- [ ] **Step 7: Commit the report**

```bash
git add docs/m4_retrieval_eval_report_*.md
git commit -m "docs: M4 retrieval-check eval report"
git push
```

- [ ] **Step 8: Close the M4 epic and surface unblocked beads**

```bash
bd close <m4-epic-id> --suggest-next
```

If a gate missed but we accept, file follow-ups (e.g., "tune retrieval key — `[h_t || z_t]` tied frame-features baseline; consider including action_summary or post-mu of next anchor") before closing.

---

## Plan self-review

Running the three checks from the writing-plans skill against the spec:

**1. Spec coverage:**

- §Vector key (`[h_t ‖ μ_q(z_t)]`, 544-dim) → Task 1 (exposes `h`) + Task 4 (`encode_game_keys`).
- §Whitening (per-dim z-score, fit on training only, params travel with index) → Task 2 + Task 6 (whitener fit inside `build_index`).
- §Corpus + leak discipline (all training anchors; exclude both holdouts' game IDs) → Task 6 (`build_index` exclusion logic) verified by Task 6 Step 1 Test 1.
- §Backend (PyTorch native, exact, batched cdist+topk on GPU) → Task 7.
- §Index artifact (`corpus_white`, `whitener`, per-row `match_id`/`minute`/`blue_win`, `checkpoint_sha`, `code_sha`, `built_at`) → Task 5 (`IndexBundle` + save/load) + Task 6 (populates the metadata fields).
- §Queries (mid-game anchors only, both holdouts, same encoder pass) → Task 12 (`_build_holdout_queries` masks by `MID_GAME_MINUTES`).
- §Metric (cohort win-rate → binary entropy) → Task 3 (`binary_entropy`) + Task 8 (`cohort_entropies`, `mean_entropy_at_k`).
- §Headline gate (`≥0.7 bits at k=64 on both holdouts`) → Task 13 (`cmd_retrieval_eval` final pass/fail line) + Task 15 Step 6 (verify).
- §k sweep (`{16, 32, 64, 128, 256}`) → constants in Task 2; sweep loop in Task 12 (`_eval_one_index`).
- §Per-minute breakdown (10..25 at headline k) → Task 11 (`per_minute_entropy_table`) + Task 12 (`headline_per_minute`).
- §Baselines (random-k, static-only, frame-features) → Task 11 (random); Task 9 (static-only); Task 10 (frame-features); Task 12 wires all three into `run_m4_eval`.
- §Self-retrieval sanity → Task 7 Step 1 Test 1 (`test_query_index_returns_self_at_rank_one_for_corpus_rows`).
- §Reporting (markdown report + HF upload of artifact) → Task 13 (`_write_report`) + Task 15 Step 5 (HF upload).
- §Files list → matches Task File-Structure section above.

No gaps.

**2. Placeholder scan:**

- Task 0 Step 2 mentions `<user>/<repo>` for an HF download path — this is a placeholder the engineer must fill from `bd memories huggingface` or by asking the user. Flagged inline ("Confirm exact HF path with the user before downloading.") so it's not a silent gap.
- Task 15 Step 8 references a `<m4-epic-id>` — the epic doesn't exist yet; the engineer creating beads for this plan would generate it. Flagged inline so it's not silent.
- All other code blocks are complete (no `TODO`, `pass`, `...`, no "fill in details").

**3. Type consistency:**

- `KEY_DIM`, `D_H`, `D_Z` defined once in `retrieval.py` and reused symbolically.
- `IndexBundle` field names (`corpus_white`, `whitener`, `row_match_id`, `row_anchor_minute`, `row_blue_win`, `checkpoint_sha`, `code_sha`, `built_at`) consistent across creation in Task 5/6/9/10 and consumption in Task 7/8/12/13.
- `encode_game_keys` returns `(keys, minutes, blue_win)` in Task 4; consumed with the same unpacking in Task 6 and Task 12.
- `query_index` signature `(bundle, queries_raw, *, k, device, batch_size) → (cohort_idx, dists)` consistent in Task 7 (definition) and Task 12 (call site).
- `cohort_entropies(cohort_idx, blue_win)` signature consistent in Task 8 (definition) and Task 12 (`_eval_one_index`).
- `MID_GAME_MINUTES`, `K_SWEEP`, `HEADLINE_K`, `HEADLINE_GATE_BITS` defined once in `retrieval.py` and imported by `m4_eval.py` and `cli.py` consistently.

No inconsistencies found.
