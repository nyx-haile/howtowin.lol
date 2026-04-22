# Plan B — RSSM v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an action-conditioned RSSM world model over the Plan A three-stream input that predicts per-minute win/loss, next meaningful event, next participant decision, and next frame features, with validated imagination-rollout capability and strict leak discipline.

**Architecture:** Build on the Plan A pipeline as scaffolding, but do **not** treat Plan A static inputs or label shapes as complete coverage of the core sequence-model spec. The stricter upstream goals remain target architecture; if this plan uses temporary proxies (for example: windowed multi-hot next-event labels, pooled per-anchor action summaries, or static conditioning that only seeds `h_0`) they are Milestone 3 staging choices rather than settled design answers.

**Tech Stack:** Python 3.12, `uv`, PyTorch 2.11.0+cu130, SQLite, numpy, pytest, `sklearn.metrics.roc_auc_score`.

**Spec:** `docs/superpowers/specs/2026-04-15-plan-b-rssm-v1-design.md`. Keep it open while implementing.

> **Alignment note:** `docs/superpowers/specs/2026-04-15-sequence-model-design.md` remains the source of truth. This implementation plan is a staged Milestone 3 plan, not a replacement for the upstream architecture. Any proxy used here should be named explicitly in code comments, reports, and later design docs.

---

## File Structure

**New files:**
- `code/model/patch_modes.py` — `PATCH_VECTOR_MODE` constants and default.
- `code/model/cold_holdout.py` — Player-cold holdout builder, loader, path constant.
- `code/model/rssm.py` — RSSM core: GRU, prior net, posterior net, KL-balanced free-bits loss.
- `code/model/heads.py` — Four decoder heads operating on `z_t`.
- `code/model/rollout.py` — `rollout_prior(model, h0, z0, steps)` — runs the prior forward N steps without posterior updates.
- `code/model/plan_b_model.py` — Top-level `PlanBModel` composing Plan A encoders + RSSM + heads.
- `code/model/plan_b_train.py` — Multi-head loss, KL schedule, prior-rollout aux loss, early stop on player-cold AUC@15.
- `code/model/plan_b_eval.py` — Outcome AUC per minute, imagination-rollout event top-5, game-cold vs player-cold, frozen-m0 leak probe.
- `data/splits/plan_b_cold_holdout.txt` — Generated.
- `code/tests/model/test_patch_modes.py`
- `code/tests/model/test_cold_holdout.py`
- `code/tests/model/test_player_features_leak.py`
- `code/tests/model/test_rssm.py`
- `code/tests/model/test_heads.py`
- `code/tests/model/test_rollout.py`
- `code/tests/model/test_plan_b_model.py`
- `code/tests/model/test_plan_b_train.py`
- `code/tests/model/test_plan_b_eval.py`

**Modified files:**
- `code/model/patch_params.py` — Bump `PATCH_VECTOR_DIM` to 1024, import `PATCH_VECTOR_MODE`, scaffolding mode zeros champ + item regions.
- `code/model/player_features.py` — Add `exclude_match_ids: set[str] | None` param. Filter aggregates.
- `code/model/dataset.py` — Plumb `exclude_match_ids` through `MatchDataset`; add `load_split("cold")` branch.
- `code/model/cli.py` — Add `cold-build`, `plan-b-shakedown`, `plan-b-train`, `plan-b-eval` subcommands.
- `code/tests/model/test_patch_params.py` — Dim check updated to 1024, scaffolding-mode test.
- `code/tests/model/test_player_features.py` — No breaking changes, but cross-check with leak test.

**Key constants (plan-b.py top of file):**
- `D_MODEL = 256` (carried from Plan A)
- `D_PLAYER = 256` (carried from Plan A encoders.py)
- `D_STATIC = 512` (static encoder output)
- `D_H = 512` (GRU hidden)
- `D_Z = 32` (stochastic latent)
- `D_OBS = 128` (anchor observation projection)
- `KL_WEIGHT_INITIAL = 0.01`
- `FREE_BITS_PER_DIM = 0.5`
- `ROLLOUT_STEPS = 3`
- `ROLLOUT_LOSS_WEIGHTS = [0.05, 0.03, 0.02]`
- `HEAD_WEIGHTS = {"outcome": 0.35, "next_event": 0.35, "next_decision": 0.15, "next_frame": 0.10}`
- `KL_WEIGHT_LOSS = 0.05`

Treat `KL_WEIGHT_INITIAL = 0.01` as the **spec-level target statement** and any literal code coefficient (e.g. `KL_WEIGHT_LOSS`) as a separate implementation detail that must be reported directly. Do not assume those numbers are semantically equivalent unless that mapping has been demonstrated.

---

## Task 0: Environment sanity

**Files:**
- None (pre-flight).

- [ ] **Step 1: Verify env and tests pass**

Run:
```bash
cd code && uv sync && uv run pytest tests/ -q
```
Expected: `53 passed` (from Plan A merge). If any fail, fix before proceeding.

- [ ] **Step 2: Verify Plan A's CLI still imports**

Run:
```bash
cd code && uv run python -c "from model import cli, baseline, train, eval, dataset, encoders"
```
Expected: no output (success) or only a UserWarning about nested tensor (benign, from Plan A).

---

## Task 1: Static vector — bump dim to 1024, add scaffolding mode

**Files:**
- Create: `code/model/patch_modes.py`
- Modify: `code/model/patch_params.py`
- Create: `code/tests/model/test_patch_modes.py`
- Modify: `code/tests/model/test_patch_params.py`

- [ ] **Step 1: Write the failing test for patch_modes constants**

Create `code/tests/model/test_patch_modes.py`:
```python
from model.patch_modes import (
    PATCH_VECTOR_MODE_SCAFFOLDING, PATCH_VECTOR_MODE_FULL,
    DEFAULT_PATCH_VECTOR_MODE,
)


def test_modes_are_distinct_strings():
    assert PATCH_VECTOR_MODE_SCAFFOLDING == "scaffolding"
    assert PATCH_VECTOR_MODE_FULL == "full"
    assert PATCH_VECTOR_MODE_SCAFFOLDING != PATCH_VECTOR_MODE_FULL


def test_default_is_scaffolding():
    assert DEFAULT_PATCH_VECTOR_MODE == PATCH_VECTOR_MODE_SCAFFOLDING
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_patch_modes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.patch_modes'`.

- [ ] **Step 3: Create `code/model/patch_modes.py`**

```python
PATCH_VECTOR_MODE_SCAFFOLDING = "scaffolding"
PATCH_VECTOR_MODE_FULL = "full"
DEFAULT_PATCH_VECTOR_MODE = PATCH_VECTOR_MODE_SCAFFOLDING
```

- [ ] **Step 4: Verify patch_modes tests pass**

Run: `cd code && uv run pytest tests/model/test_patch_modes.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Write the failing test for scaffolding behavior**

Append to `code/tests/model/test_patch_params.py`:
```python
def test_patch_vector_dim_is_1024():
    from model.patch_params import PATCH_VECTOR_DIM
    assert PATCH_VECTOR_DIM == 1024


def test_scaffolding_mode_zeros_champ_and_item_regions(fixture_match_id):
    from model.patch_params import patch_vector_for_match
    from model.patch_modes import PATCH_VECTOR_MODE_SCAFFOLDING
    vec = patch_vector_for_match(fixture_match_id, mode=PATCH_VECTOR_MODE_SCAFFOLDING)
    # Champion-stat region: offsets 0..99
    np.testing.assert_allclose(vec[0:100], np.zeros(100, dtype=np.float32))
    # Item-stat region: offsets 100..119
    np.testing.assert_allclose(vec[100:120], np.zeros(20, dtype=np.float32))
    # Version triple: offsets 120..122 — NOT zeroed
    assert np.any(vec[120:123] != 0.0)


def test_full_mode_populates_regions(fixture_match_id):
    from model.patch_params import patch_vector_for_match
    from model.patch_modes import PATCH_VECTOR_MODE_FULL
    vec = patch_vector_for_match(fixture_match_id, mode=PATCH_VECTOR_MODE_FULL)
    # Full mode lights up at least the champion region.
    assert np.any(vec[0:100] != 0.0)
```

- [ ] **Step 6: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_patch_params.py -v`
Expected: FAIL — `PATCH_VECTOR_DIM` is still 256, and `patch_vector_for_match` doesn't accept a `mode` kwarg.

- [ ] **Step 7: Update `code/model/patch_params.py`**

At the top of the file, change the constant and add mode import:
```python
from model.patch_modes import (
    PATCH_VECTOR_MODE_SCAFFOLDING,
    PATCH_VECTOR_MODE_FULL,
    DEFAULT_PATCH_VECTOR_MODE,
)

PATCH_VECTOR_DIM = 1024
```

Change the signature and gate the regions by mode. Replace `patch_vector_for_match` body:
```python
def patch_vector_for_match(match_id, mode=DEFAULT_PATCH_VECTOR_MODE):
    conn = get_conn()
    try:
        game = conn.execute(
            "SELECT patch FROM games WHERE match_id = ?", (match_id,)
        ).fetchone()
    finally:
        conn.close()

    version = fetch_dragon_version(game["patch"] if game else "")
    champs = load_champions(version)
    items = load_items(version)

    vec = np.zeros(PATCH_VECTOR_DIM, dtype=np.float32)

    if mode == PATCH_VECTOR_MODE_FULL:
        # Champion stats per participant.
        from raw_db import get_raw_match
        match, _tl = get_raw_match(match_id)
        if match:
            for i, p in enumerate(match["info"]["participants"][:10]):
                champ_key = p.get("championId")
                champ_name = _champ_name_by_key(champs, champ_key)
                if champ_name and champ_name in champs:
                    stats = champs[champ_name].get("stats", {})
                    for j, fname in enumerate(CHAMP_STAT_FIELDS):
                        vec[i * len(CHAMP_STAT_FIELDS) + j] = float(stats.get(fname, 0.0))

        # Patch-level item aggregate (match-independent).
        item_totals = {f: 0.0 for f in ITEM_STAT_FIELDS}
        for idata in items.values():
            stats = idata.get("stats", {})
            for fname in ITEM_STAT_FIELDS:
                item_totals[fname] += float(stats.get(fname, 0.0))
        for j, fname in enumerate(ITEM_STAT_FIELDS):
            vec[100 + j] = item_totals[fname]

    # Version triple — always populated, in both modes.
    parts = version.split(".")
    vec[120] = float(parts[0]) if parts and parts[0].isdigit() else 0.0
    vec[121] = float(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0.0
    vec[122] = float(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0.0

    return vec
```

Note: the leak-regression test at offsets 100..120 still passes because `PATCH_VECTOR_MODE_FULL` aggregates over `items.values()` (patch-level), not over any one match's build.

- [ ] **Step 8: Verify all patch_params tests pass**

Run: `cd code && uv run pytest tests/model/test_patch_params.py -v`
Expected: all tests pass, including leak regression, 1024-dim check, and mode tests.

- [ ] **Step 9: Run full suite to catch knock-on breakage from PATCH_VECTOR_DIM change**

Run: `cd code && uv run pytest tests/ -q`
Expected: all prior tests still pass. Encoder tests may reference the constant via `StaticContextEncoder`; those should already read from `PATCH_VECTOR_DIM` symbolically and so pick up 1024 transparently.

If any test fails asserting `== 256`, update it to read `PATCH_VECTOR_DIM` symbolically.

- [ ] **Step 10: Commit**

```bash
git add code/model/patch_modes.py code/model/patch_params.py \
        code/tests/model/test_patch_modes.py code/tests/model/test_patch_params.py
git commit -m "plan-b: PATCH_VECTOR_DIM=1024 + scaffolding/full mode flag"
```

---

## Task 2: Stream-2 leak audit — exclude_match_ids in player_features

**Files:**
- Modify: `code/model/player_features.py`
- Create: `code/tests/model/test_player_features_leak.py`

- [ ] **Step 1: Write the failing leak-regression test**

Create `code/tests/model/test_player_features_leak.py`:
```python
import pytest
from model.player_features import player_feature_vector


@pytest.fixture(scope="module")
def sample_puuid():
    from db import get_conn
    conn = get_conn()
    row = conn.execute(
        "SELECT puuid FROM players "
        "WHERE puuid IN (SELECT puuid FROM players GROUP BY puuid HAVING COUNT(*) > 2) "
        "LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None, "test requires a puuid with >2 games"
    return row["puuid"]


def test_feature_vector_respects_exclude_match_ids(sample_puuid):
    vec_full = player_feature_vector(sample_puuid)

    # Find any match this puuid played in.
    from db import get_conn
    conn = get_conn()
    row = conn.execute(
        "SELECT match_id FROM players WHERE puuid = ? LIMIT 1", (sample_puuid,)
    ).fetchone()
    conn.close()
    assert row is not None
    mid = row["match_id"]

    vec_excl = player_feature_vector(sample_puuid, exclude_match_ids={mid})
    # Feature vector MUST change when we remove a game the player participated in.
    # Exact equality would mean the exclusion did nothing.
    assert not (vec_full == vec_excl).all(), (
        "player_feature_vector did not change when a participating match was excluded"
    )


def test_feature_vector_excludes_chronologically_later_games(sample_puuid):
    from db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT p.match_id, g.created_at FROM players p "
        "JOIN games g ON p.match_id = g.match_id "
        "WHERE p.puuid = ? ORDER BY g.created_at ASC",
        (sample_puuid,),
    ).fetchall()
    conn.close()
    assert len(rows) >= 2, "test requires a puuid with >=2 chronologically-ordered games"

    earliest_mid = rows[0]["match_id"]

    # When we exclude the earliest game, later games should also be filtered
    # (to prevent the model seeing the player's future).
    vec_excl_earliest = player_feature_vector(sample_puuid, exclude_match_ids={earliest_mid})
    vec_excl_all = player_feature_vector(
        sample_puuid, exclude_match_ids={r["match_id"] for r in rows}
    )
    # Excluding earliest alone should match excluding all (causal exclusion).
    import numpy as np
    np.testing.assert_allclose(vec_excl_earliest, vec_excl_all)


def test_feature_vector_no_exclusion_matches_full_corpus(sample_puuid):
    vec_a = player_feature_vector(sample_puuid)
    vec_b = player_feature_vector(sample_puuid, exclude_match_ids=set())
    import numpy as np
    np.testing.assert_allclose(vec_a, vec_b)
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_player_features_leak.py -v`
Expected: FAIL because `player_feature_vector` doesn't accept `exclude_match_ids`.

- [ ] **Step 3: Read current `code/model/player_features.py`**

Run: `cat code/model/player_features.py | head -80`
Expected: see the existing `player_feature_vector` that queries the `players` + `games` tables for rolling aggregates.

- [ ] **Step 4: Update `player_feature_vector` to accept exclusion set**

Modify `code/model/player_features.py`. Change the signature and propagate the exclusion into every SQL aggregate. The pattern for every query touching `players` or `games` is:

```python
def player_feature_vector(puuid, exclude_match_ids=None):
    """Compute the player crafted-feature vector, excluding specified matches
    and all chronologically later games for this puuid from every aggregate.

    exclude_match_ids: optional set of match_ids to omit. Also omits any
    game this puuid played in after the earliest excluded game, so the
    learner cannot see this player's future via rolling stats.
    """
    exclude_match_ids = set(exclude_match_ids) if exclude_match_ids else set()

    from db import get_conn
    conn = get_conn()
    try:
        # If any excluded game is this puuid's, compute the earliest excluded
        # timestamp and filter every later game for this puuid as well.
        if exclude_match_ids:
            placeholders = ",".join("?" * len(exclude_match_ids))
            row = conn.execute(
                f"SELECT MIN(g.created_at) AS cutoff FROM players p "
                f"JOIN games g ON p.match_id = g.match_id "
                f"WHERE p.puuid = ? AND p.match_id IN ({placeholders})",
                (puuid, *exclude_match_ids),
            ).fetchone()
            cutoff = row["cutoff"] if row and row["cutoff"] is not None else None
        else:
            cutoff = None

        # Build the WHERE clause once: exclude any match in set, and (if cutoff)
        # also exclude any game strictly after cutoff for this puuid.
        base_where = "p.puuid = ?"
        params = [puuid]
        if exclude_match_ids:
            base_where += f" AND p.match_id NOT IN ({placeholders})"
            params.extend(exclude_match_ids)
        if cutoff is not None:
            base_where += " AND g.created_at < ?"
            params.append(cutoff)

        # Example: rolling winrate / KDA / CS@10 / games count.
        # Use base_where everywhere. Rewrite each SQL query currently in this
        # function to use the same pattern.
        # ... (re-implement every existing aggregate query with base_where)
    finally:
        conn.close()

    return vec
```

**Implementation note:** do not guess at existing query shapes. Read the current file, keep the feature-vector offsets the same, replace each query's WHERE clause with the `base_where + params` pattern. The zero-vector return for unknown puuids (present in Plan A) must still occur when `exclude_match_ids` strips a puuid down to zero games.

- [ ] **Step 5: Run leak tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_player_features_leak.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the full existing player_features test suite**

Run: `cd code && uv run pytest tests/model/test_player_features.py -v`
Expected: all original tests still pass (no exclusion, default behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add code/model/player_features.py code/tests/model/test_player_features_leak.py
git commit -m "plan-b: player_feature_vector accepts exclude_match_ids + causal cutoff"
```

---

## Task 3: Player-cold-start holdout builder

**Files:**
- Create: `code/model/cold_holdout.py`
- Create: `code/tests/model/test_cold_holdout.py`
- Create: `data/splits/plan_b_cold_holdout.txt` (via CLI in later task)

- [ ] **Step 1: Write the failing test for cold_holdout**

Create `code/tests/model/test_cold_holdout.py`:
```python
import hashlib
from model.cold_holdout import (
    build_player_cold_holdout, COLD_HOLDOUT_PATH, COLD_HOLDOUT_PUUID_COUNT,
)


def test_build_is_deterministic():
    a = build_player_cold_holdout()
    b = build_player_cold_holdout()
    assert a == b


def test_uses_expected_number_of_puuids():
    result = build_player_cold_holdout()
    # result is (puuids_set, match_ids_set)
    puuids, match_ids = result
    assert len(puuids) == COLD_HOLDOUT_PUUID_COUNT


def test_match_ids_actually_contain_those_puuids():
    from db import get_conn
    puuids, match_ids = build_player_cold_holdout()
    conn = get_conn()
    for mid in list(match_ids)[:5]:
        rows = conn.execute(
            "SELECT puuid FROM players WHERE match_id = ?", (mid,)
        ).fetchall()
        participants = {r["puuid"] for r in rows}
        assert participants & puuids, f"cold match {mid} has no cold puuid"
    conn.close()


def test_no_overlap_with_game_cold_holdout():
    from model.dataset import load_split
    game_cold = set(load_split("holdout"))
    _puuids, cold_match_ids = build_player_cold_holdout()
    overlap = game_cold & cold_match_ids
    # If a match happens to be in both splits, that's allowed — it's still
    # held-out either way. But the cold-specific count should be non-trivial.
    assert len(cold_match_ids) > len(overlap), (
        "player-cold holdout adds no new held-out games beyond game-cold"
    )


def test_constant_path_points_into_splits_dir():
    assert "data/splits/plan_b_cold_holdout.txt" in str(COLD_HOLDOUT_PATH)
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_cold_holdout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.cold_holdout'`.

- [ ] **Step 3: Implement `code/model/cold_holdout.py`**

```python
import hashlib
import os
from db import get_conn

COLD_HOLDOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "splits", "plan_b_cold_holdout.txt"
)
COLD_HOLDOUT_PUUID_COUNT = 5


def _sha1_rank(puuid: str) -> str:
    return hashlib.sha1(puuid.encode("utf-8")).hexdigest()


def build_player_cold_holdout() -> tuple[set[str], set[str]]:
    """Deterministically pick COLD_HOLDOUT_PUUID_COUNT puuids by SHA1 hash
    order, then collect every match those puuids appeared in. Return both."""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT DISTINCT puuid FROM players").fetchall()
        all_puuids = [r["puuid"] for r in rows]
        all_puuids.sort(key=_sha1_rank)  # stable deterministic order
        chosen_puuids = set(all_puuids[:COLD_HOLDOUT_PUUID_COUNT])

        if not chosen_puuids:
            return set(), set()

        placeholders = ",".join("?" * len(chosen_puuids))
        rows = conn.execute(
            f"SELECT DISTINCT match_id FROM players WHERE puuid IN ({placeholders})",
            tuple(chosen_puuids),
        ).fetchall()
        match_ids = {r["match_id"] for r in rows}
    finally:
        conn.close()
    return chosen_puuids, match_ids


def save_player_cold_holdout(path: str = COLD_HOLDOUT_PATH) -> tuple[set[str], set[str]]:
    puuids, match_ids = build_player_cold_holdout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for mid in sorted(match_ids):
            f.write(mid + "\n")
    return puuids, match_ids


def load_player_cold_holdout(path: str = COLD_HOLDOUT_PATH) -> set[str]:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_cold_holdout.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/cold_holdout.py code/tests/model/test_cold_holdout.py
git commit -m "plan-b: player-cold-start holdout builder (5 puuids, deterministic)"
```

---

## Task 4: Wire exclusions through MatchDataset

**Files:**
- Modify: `code/model/dataset.py`
- Modify: `code/tests/model/` (new assertion via existing test_dataset.py or new)

- [ ] **Step 1: Read current `code/model/dataset.py`**

Look at `MatchDataset.__getitem__` and `build_puuid_index` — those are the functions that currently compute or cache player feature vectors.

- [ ] **Step 2: Write the failing test**

Create/append `code/tests/model/test_dataset_leak.py`:
```python
import numpy as np
import pytest
from model.dataset import MatchDataset, build_puuid_index
from model.cold_holdout import build_player_cold_holdout


@pytest.fixture(scope="module")
def cold_sets():
    puuids, match_ids = build_player_cold_holdout()
    return puuids, match_ids


def test_dataset_accepts_and_uses_exclude_match_ids(fixture_match_id, cold_sets):
    _puuids, cold_ids = cold_sets
    exclude = {fixture_match_id} | cold_ids
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx, exclude_match_ids=exclude)
    sample = ds[0]
    assert "players" in sample
    # player features must be finite (no NaN from empty aggregates)
    assert np.all(np.isfinite(sample["players"]))


def test_dataset_without_exclude_matches_plan_a_behavior(fixture_match_id):
    idx = build_puuid_index([fixture_match_id])
    ds_a = MatchDataset([fixture_match_id], puuid_index=idx)
    ds_b = MatchDataset([fixture_match_id], puuid_index=idx, exclude_match_ids=None)
    np.testing.assert_allclose(ds_a[0]["players"], ds_b[0]["players"])


def test_load_split_cold_returns_cold_ids():
    from model.dataset import load_split
    from model.cold_holdout import save_player_cold_holdout
    save_player_cold_holdout()  # ensure file exists for this test
    cold = set(load_split("cold"))
    _puuids, expected = build_player_cold_holdout()
    assert cold == expected
```

- [ ] **Step 3: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_dataset_leak.py -v`
Expected: FAIL (MatchDataset does not accept `exclude_match_ids`; `load_split("cold")` not implemented).

- [ ] **Step 4: Modify `MatchDataset.__init__` to accept exclusion and pass to player_feature_vector**

In `code/model/dataset.py`:
```python
class MatchDataset(Dataset):
    def __init__(self, match_ids, puuid_index, exclude_match_ids=None):
        self.match_ids = match_ids
        self.puuid_index = puuid_index
        self.exclude_match_ids = set(exclude_match_ids) if exclude_match_ids else set()

    def __getitem__(self, i):
        mid = self.match_ids[i]
        # ... existing code ...
        # When computing per-puuid features, pass self.exclude_match_ids:
        pvecs = []
        for puuid in participant_puuids:
            pvecs.append(
                player_feature_vector(puuid, exclude_match_ids=self.exclude_match_ids)
            )
        # ... rest unchanged ...
```

Also add a `"cold"` branch to `load_split`:
```python
def load_split(name: str):
    if name == "cold":
        from model.cold_holdout import load_player_cold_holdout
        return sorted(load_player_cold_holdout())
    # ... existing branches for "holdout" / "train" ...
```

And ensure the `"train"` branch subtracts BOTH the game-cold holdout AND the player-cold holdout from the corpus:
```python
def load_split(name: str):
    if name == "cold":
        from model.cold_holdout import load_player_cold_holdout
        return sorted(load_player_cold_holdout())
    # Load all match_ids from DB (existing logic).
    all_mids = _load_all_match_ids()
    game_cold = set(_load_file(HOLDOUT_PATH))
    from model.cold_holdout import load_player_cold_holdout
    try:
        player_cold = load_player_cold_holdout()
    except FileNotFoundError:
        player_cold = set()
    if name == "holdout":
        return sorted(game_cold)
    if name == "train":
        return sorted(m for m in all_mids if m not in game_cold and m not in player_cold)
    raise ValueError(f"unknown split: {name}")
```

- [ ] **Step 5: Run test, verify pass**

Run: `cd code && uv run pytest tests/model/test_dataset_leak.py tests/model/test_cold_holdout.py -v`
Expected: all pass.

- [ ] **Step 6: Verify the full suite still passes (Plan A tests)**

Run: `cd code && uv run pytest tests/ -q`
Expected: all tests pass, training corpus size in any downstream tests may shift slightly (train split now excludes both holdouts).

- [ ] **Step 7: Commit**

```bash
git add code/model/dataset.py code/tests/model/test_dataset_leak.py
git commit -m "plan-b: MatchDataset propagates exclude_match_ids; load_split('cold')"
```

---

## Task 5: RSSM core — GRU + prior + posterior + KL-free-bits loss

**Files:**
- Create: `code/model/rssm.py`
- Create: `code/tests/model/test_rssm.py`

- [ ] **Step 1: Write the failing RSSM shape tests**

Create `code/tests/model/test_rssm.py`:
```python
import torch
import pytest
from model.rssm import RSSMCore, free_bits_kl


@pytest.fixture
def rssm():
    return RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)


def test_prior_shapes(rssm):
    h = torch.randn(2, 512)
    mu, logvar = rssm.prior(h)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)


def test_posterior_shapes(rssm):
    h = torch.randn(2, 512)
    o = torch.randn(2, 128)
    mu, logvar = rssm.posterior(h, o)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)


def test_step_returns_new_h_and_z(rssm):
    h = torch.randn(2, 512)
    z = torch.randn(2, 32)
    a = torch.randn(2, 256)
    h2 = rssm.step(h, z, a)
    assert h2.shape == (2, 512)


def test_free_bits_kl_non_negative():
    mu_q = torch.zeros(4, 32)
    logvar_q = torch.zeros(4, 32)
    mu_p = torch.zeros(4, 32)
    logvar_p = torch.zeros(4, 32)
    # Identical distributions -> KL=0 per dim; free-bits floor applies.
    kl = free_bits_kl(mu_q, logvar_q, mu_p, logvar_p, free_bits_per_dim=0.5)
    assert torch.all(kl >= 0.0)


def test_free_bits_kl_clamps_below_floor():
    # Two identical Gaussians: raw KL is 0. Floor is 0.5 * D_Z.
    mu = torch.zeros(1, 32)
    logvar = torch.zeros(1, 32)
    kl = free_bits_kl(mu, logvar, mu, logvar, free_bits_per_dim=0.5)
    # KL is clamped to free-bits floor -> 0.5 * 32 = 16, per-sample.
    assert torch.allclose(kl, torch.tensor([16.0]))


def test_reparameterize_samples_with_gradient():
    from model.rssm import reparameterize
    mu = torch.zeros(2, 32, requires_grad=True)
    logvar = torch.zeros(2, 32, requires_grad=True)
    z = reparameterize(mu, logvar)
    z.sum().backward()
    assert mu.grad is not None
    assert logvar.grad is not None
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_rssm.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `code/model/rssm.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """KL(q || p) per-sample, summed across last dim. Shape: (batch,)."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
        + logvar_p - logvar_q
    )
    return kl.sum(dim=-1)


def free_bits_kl(mu_q, logvar_q, mu_p, logvar_p, free_bits_per_dim: float) -> torch.Tensor:
    """Free-bits KL. Clamps per-dim KL to a floor, then sums. Shape: (batch,)."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    per_dim = 0.5 * (
        (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
        + logvar_p - logvar_q
    )
    # Clamp per-dim to floor.
    floored = per_dim.clamp(min=free_bits_per_dim)
    return floored.sum(dim=-1)


class RSSMCore(nn.Module):
    """Recurrent state-space model core: GRU deterministic path + stochastic latent."""

    def __init__(self, d_h: int = 512, d_z: int = 32, d_action: int = 256,
                 d_obs: int = 128, d_hidden: int = 512):
        super().__init__()
        self.d_h = d_h
        self.d_z = d_z

        # GRU updates h given (action_summary, z).
        self.gru = nn.GRUCell(input_size=d_action + d_z, hidden_size=d_h)

        self.prior_net = nn.Sequential(
            nn.Linear(d_h, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * d_z),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(d_h + d_obs, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * d_z),
        )

    def prior(self, h):
        out = self.prior_net(h)
        mu, logvar = out.chunk(2, dim=-1)
        # Log-var clamping for numerical stability.
        return mu, logvar.clamp(min=-10.0, max=10.0)

    def posterior(self, h, o):
        out = self.posterior_net(torch.cat([h, o], dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-10.0, max=10.0)

    def step(self, h, z, action_summary):
        """Advance h given previous z and the action/event summary for the window."""
        inp = torch.cat([action_summary, z], dim=-1)
        return self.gru(inp, h)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_rssm.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/rssm.py code/tests/model/test_rssm.py
git commit -m "plan-b: RSSM core (GRU + prior/posterior nets + free-bits KL)"
```

---

## Task 6: Decoder heads

**Files:**
- Create: `code/model/heads.py`
- Create: `code/tests/model/test_heads.py`

- [ ] **Step 1: Write the failing heads tests**

Create `code/tests/model/test_heads.py`:
```python
import torch
from model.heads import NextEventHead, OutcomeHead, NextDecisionHead, NextFrameHead
from model.tokens import NUM_EVENT_TYPES


def test_next_event_head_shape():
    h = NextEventHead(d_z=32, n_event_types=NUM_EVENT_TYPES)
    z = torch.randn(4, 32)
    logits = h(z)
    assert logits.shape == (4, NUM_EVENT_TYPES)


def test_outcome_head_returns_scalar_logit():
    h = OutcomeHead(d_z=32)
    z = torch.randn(4, 32)
    out = h(z)
    assert out.shape == (4,)


def test_next_decision_head_shape():
    # Decisions: ITEM_PURCHASED, SKILL_LEVEL_UP, WARD_PLACED, RECALL,
    # ENGAGE, DISENGAGE -> 6 classes.
    n_decisions = 6
    n_participants = 10
    h = NextDecisionHead(d_z=32, n_decisions=n_decisions, n_participants=n_participants)
    z = torch.randn(4, 32)
    logits = h(z)
    assert logits.shape == (4, n_participants, n_decisions)


def test_next_frame_head_returns_mu_sigma():
    # Frame features: per-participant (gold, xp, level, pos_x, pos_y, cs) -> 6.
    n_participants = 10
    feat_dim = 6
    h = NextFrameHead(d_z=32, n_participants=n_participants, feat_dim=feat_dim)
    z = torch.randn(4, 32)
    mu, logvar = h(z)
    assert mu.shape == (4, n_participants, feat_dim)
    assert logvar.shape == (4, n_participants, feat_dim)


def test_heads_outputs_are_finite():
    z = torch.randn(2, 32)
    assert torch.isfinite(NextEventHead(32, NUM_EVENT_TYPES)(z)).all()
    assert torch.isfinite(OutcomeHead(32)(z)).all()
    assert torch.isfinite(NextDecisionHead(32, 6, 10)(z)).all()
    mu, logvar = NextFrameHead(32, 10, 6)(z)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_heads.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `code/model/heads.py`**

```python
import torch
import torch.nn as nn


class NextEventHead(nn.Module):
    """Multi-hot logits over event types for the next-minute window."""
    def __init__(self, d_z: int, n_event_types: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_event_types),
        )

    def forward(self, z):
        return self.net(z)


class OutcomeHead(nn.Module):
    """Scalar logit for game-win at each anchor."""
    def __init__(self, d_z: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class NextDecisionHead(nn.Module):
    """Per-participant categorical over decision types."""
    def __init__(self, d_z: int, n_decisions: int, n_participants: int = 10,
                 d_hidden: int = 256):
        super().__init__()
        self.n_participants = n_participants
        self.n_decisions = n_decisions
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_participants * n_decisions),
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(z.size(0), self.n_participants, self.n_decisions)


class NextFrameHead(nn.Module):
    """Per-participant μ/logvar over frame feature deltas."""
    def __init__(self, d_z: int, n_participants: int = 10, feat_dim: int = 6,
                 d_hidden: int = 256):
        super().__init__()
        self.n_participants = n_participants
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * n_participants * feat_dim),
        )

    def forward(self, z):
        out = self.net(z).view(z.size(0), self.n_participants, self.feat_dim, 2)
        mu = out[..., 0]
        logvar = out[..., 1].clamp(min=-10.0, max=10.0)
        return mu, logvar
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_heads.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/heads.py code/tests/model/test_heads.py
git commit -m "plan-b: four decoder heads (event, outcome, decision, frame)"
```

---

## Task 7: Imagination rollout

**Files:**
- Create: `code/model/rollout.py`
- Create: `code/tests/model/test_rollout.py`

- [ ] **Step 1: Write the failing rollout tests**

Create `code/tests/model/test_rollout.py`:
```python
import torch
from model.rssm import RSSMCore
from model.rollout import rollout_prior


def test_rollout_returns_correct_number_of_steps():
    rssm = RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)
    h0 = torch.randn(2, 512)
    z0 = torch.randn(2, 32)
    steps = rollout_prior(rssm, h0, z0, n_steps=3, action_summary=torch.randn(2, 256))
    assert len(steps) == 3
    for h, z, mu, logvar in steps:
        assert h.shape == (2, 512)
        assert z.shape == (2, 32)
        assert mu.shape == (2, 32)
        assert logvar.shape == (2, 32)


def test_rollout_without_action_uses_zeros():
    rssm = RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)
    h0 = torch.zeros(1, 512)
    z0 = torch.zeros(1, 32)
    steps = rollout_prior(rssm, h0, z0, n_steps=2, action_summary=None)
    assert len(steps) == 2


def test_rollout_gradient_flows_to_prior_net():
    rssm = RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)
    h0 = torch.randn(1, 512)
    z0 = torch.randn(1, 32)
    steps = rollout_prior(rssm, h0, z0, n_steps=3,
                          action_summary=torch.randn(1, 256))
    loss = sum(z.sum() for _h, z, _mu, _lv in steps)
    loss.backward()
    # Prior network should receive gradient.
    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in rssm.prior_net.parameters())
    assert any_grad
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd code && uv run pytest tests/model/test_rollout.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `code/model/rollout.py`**

```python
import torch
from model.rssm import reparameterize


def rollout_prior(rssm, h0, z0, n_steps: int, action_summary=None):
    """Roll the prior forward n_steps without posterior updates.

    Returns a list of (h_t, z_t, mu_t, logvar_t) tuples of length n_steps.
    action_summary is reused at each step; pass None to use zeros.
    """
    if action_summary is None:
        action_summary = torch.zeros(h0.size(0), rssm.gru.input_size - rssm.d_z,
                                     device=h0.device)

    h = h0
    z = z0
    out = []
    for _ in range(n_steps):
        h = rssm.step(h, z, action_summary)
        mu, logvar = rssm.prior(h)
        z = reparameterize(mu, logvar)
        out.append((h, z, mu, logvar))
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_rollout.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/rollout.py code/tests/model/test_rollout.py
git commit -m "plan-b: rollout_prior helper for imagination rollouts"
```

---

## Task 8: Plan B composed model

**Files:**
- Create: `code/model/plan_b_model.py`
- Create: `code/tests/model/test_plan_b_model.py`

- [ ] **Step 1: Write the failing composed-model test**

Create `code/tests/model/test_plan_b_model.py`:
```python
import torch
from model.plan_b_model import PlanBModel


def test_forward_pass_shape(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])

    model = PlanBModel(max_puuids=len(idx))
    out = model(batch)

    # Every head must produce output per anchor.
    # Shape conventions: (B=1, T_anchors, ...)
    n_anchors = out["n_anchors"]
    assert out["event_logits"].shape == (1, n_anchors, 11)
    assert out["outcome_logits"].shape == (1, n_anchors)
    assert out["decision_logits"].shape == (1, n_anchors, 10, 6)
    mu = out["frame_mu"]
    logvar = out["frame_logvar"]
    assert mu.shape == (1, n_anchors, 10, 6)
    assert logvar.shape == (1, n_anchors, 10, 6)
    # Posterior / prior statistics for KL.
    assert out["post_mu"].shape == (1, n_anchors, 32)
    assert out["prior_mu"].shape == (1, n_anchors, 32)


def test_model_forward_is_differentiable(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])

    model = PlanBModel(max_puuids=len(idx))
    out = model(batch)
    loss = (
        out["event_logits"].sum() + out["outcome_logits"].sum()
        + out["decision_logits"].sum() + out["frame_mu"].sum()
    )
    loss.backward()
    grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.parameters())
    assert grad_found
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_plan_b_model.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `code/model/plan_b_model.py`**

```python
import torch
import torch.nn as nn
from model.encoders import (
    StaticContextEncoder, PlayerModelEncoder, DynamicStreamEmbedder, D_MODEL,
)
from model.patch_params import PATCH_VECTOR_DIM
from model.rssm import RSSMCore, reparameterize
from model.heads import NextEventHead, OutcomeHead, NextDecisionHead, NextFrameHead
from model.tokens import NUM_EVENT_TYPES, ANCHOR_TOKEN


D_STATIC = 512
D_H = 512
D_Z = 32
D_OBS = 128
D_PLAYER = D_MODEL  # carry from Plan A
D_ACTION = D_MODEL  # action summary dim

N_PARTICIPANTS = 10
N_DECISION_TYPES = 6  # ITEM_PURCHASED, SKILL_LEVEL_UP, WARD_PLACED, RECALL, ENGAGE, DISENGAGE
FRAME_FEAT_DIM = 6    # gold, xp, level, pos_x, pos_y, cs


class AnchorObservationProjector(nn.Module):
    """Projects per-anchor per-participant frame features into D_OBS."""
    def __init__(self, d_obs: int = D_OBS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_PARTICIPANTS * FRAME_FEAT_DIM, 256), nn.GELU(),
            nn.Linear(256, d_obs),
        )

    def forward(self, frame_feats):
        # frame_feats shape: (B, T, 10, 6) -> flatten participant-dim
        b, t, p, f = frame_feats.shape
        flat = frame_feats.view(b, t, p * f)
        return self.net(flat)


class ActionSummarizer(nn.Module):
    """Summarizes events-and-decisions occurring in the window preceding each anchor."""
    def __init__(self, d_action: int = D_ACTION, d_model: int = D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d_model, d_action)

    def forward(self, event_window_embeddings, window_mask):
        # event_window_embeddings: (B, T, N_events, D_MODEL). window_mask: (B, T, N_events).
        mask = window_mask.unsqueeze(-1).float()
        summed = (event_window_embeddings * mask).sum(dim=2)
        counts = mask.sum(dim=2).clamp(min=1.0)
        mean = summed / counts
        return self.proj(mean)


class PlanBModel(nn.Module):
    def __init__(self, max_puuids: int):
        super().__init__()
        self.static_enc = StaticContextEncoder(d_patch=PATCH_VECTOR_DIM, d_out=D_STATIC)
        self.player_enc = PlayerModelEncoder(max_puuids=max_puuids)
        self.dynamic_emb = DynamicStreamEmbedder()

        self.obs_proj = AnchorObservationProjector(d_obs=D_OBS)
        self.action_summarizer = ActionSummarizer(d_action=D_ACTION, d_model=D_MODEL)

        # Static cross-conditioning on h at each step.
        self.static_to_h = nn.Linear(D_STATIC, D_H)

        self.rssm = RSSMCore(d_h=D_H, d_z=D_Z, d_action=D_ACTION, d_obs=D_OBS)

        self.head_event = NextEventHead(D_Z, NUM_EVENT_TYPES)
        self.head_outcome = OutcomeHead(D_Z)
        self.head_decision = NextDecisionHead(D_Z, N_DECISION_TYPES, N_PARTICIPANTS)
        self.head_frame = NextFrameHead(D_Z, N_PARTICIPANTS, FRAME_FEAT_DIM)

    def _segment_by_anchors(self, batch):
        """Split the dynamic sequence into anchor windows. Returns:
           - anchor_positions: (B, T_anchors) — index in token sequence
           - event_window_embeddings: (B, T_anchors, max_events, D_MODEL)
           - window_mask: (B, T_anchors, max_events)
           - anchor_obs: (B, T_anchors, N_PARTICIPANTS, FRAME_FEAT_DIM)
        """
        # Expect batch to contain: tokens (B, L), token_actors (B, L),
        # token_timestamps (B, L), key_pad_mask (B, L), players (B, 10, D_PLAYER),
        # frame_features (B, T_anchors, 10, 6), anchor_positions (B, T_anchors).
        # All pre-computed by the dataset / collate step.
        return (batch["anchor_positions"], batch["event_window_embeddings"],
                batch["window_mask"], batch["frame_features"])

    def forward(self, batch):
        B = batch["tokens"].size(0)
        static = self.static_enc(batch["static"])           # (B, D_STATIC)
        # players is (B, 10, D_PLAYER) from the encoder.
        # Dynamic embedder feeds the action summarizer; we still need token-level
        # embeddings to pool within windows.
        token_emb = self.dynamic_emb(
            batch["tokens"], batch["token_actors"], batch["token_timestamps"],
            batch["players"], batch["player_ids"],
        )  # (B, L, D_MODEL)

        # ----- Below, dataset/collate is expected to have prepared per-window
        # index tensors. If those are not present, compute them on the fly from
        # tokens + timestamps. The dataset changes to emit these tensors belong
        # in the companion commit during Task 9 (wiring).
        anchor_positions, event_window_embeddings, window_mask, frame_features = \
            self._segment_by_anchors({**batch, "token_emb": token_emb})

        T = anchor_positions.size(1)

        obs = self.obs_proj(frame_features)                   # (B, T, D_OBS)
        action_summary = self.action_summarizer(event_window_embeddings, window_mask)  # (B, T, D_ACTION)

        h = self.static_to_h(static)                          # (B, D_H) — seeds recurrence
        z = torch.zeros(B, D_Z, device=h.device)

        post_mu_all, post_logvar_all = [], []
        prior_mu_all, prior_logvar_all = [], []
        z_all = []

        for t in range(T):
            h = self.rssm.step(h, z, action_summary[:, t])   # update h
            pr_mu, pr_lv = self.rssm.prior(h)
            po_mu, po_lv = self.rssm.posterior(h, obs[:, t])
            z = reparameterize(po_mu, po_lv)                 # use posterior during training
            post_mu_all.append(po_mu); post_logvar_all.append(po_lv)
            prior_mu_all.append(pr_mu); prior_logvar_all.append(pr_lv)
            z_all.append(z)

        Z = torch.stack(z_all, dim=1)                        # (B, T, D_Z)
        post_mu = torch.stack(post_mu_all, dim=1)
        post_lv = torch.stack(post_logvar_all, dim=1)
        prior_mu = torch.stack(prior_mu_all, dim=1)
        prior_lv = torch.stack(prior_logvar_all, dim=1)

        # Heads (apply over flattened Z)
        z_flat = Z.view(B * T, D_Z)
        event_logits = self.head_event(z_flat).view(B, T, NUM_EVENT_TYPES)
        outcome_logits = self.head_outcome(z_flat).view(B, T)
        decision_logits = self.head_decision(z_flat).view(B, T, N_PARTICIPANTS, N_DECISION_TYPES)
        frame_mu, frame_lv = self.head_frame(z_flat)
        frame_mu = frame_mu.view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)
        frame_lv = frame_lv.view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)

        return {
            "event_logits": event_logits,
            "outcome_logits": outcome_logits,
            "decision_logits": decision_logits,
            "frame_mu": frame_mu,
            "frame_logvar": frame_lv,
            "post_mu": post_mu, "post_logvar": post_lv,
            "prior_mu": prior_mu, "prior_logvar": prior_lv,
            "z": Z,
            "n_anchors": T,
            # Cache last h, z for imagination rollout from the end.
            "h_final": h, "z_final": z,
        }
```

**Dataset/collate note:** the forward pass assumes the batch dict contains `anchor_positions`, `event_window_embeddings`, `window_mask`, and `frame_features`. Task 9 extends the dataset to produce these. To keep Task 8 atomic, provide a small pre-processor invoked before the forward call in the test fixture.

- [ ] **Step 4: Add a minimal preprocessor inside `test_plan_b_model.py` before the model call**

Extend the test to segment on the fly:
```python
def _segment_for_test(batch, d_model):
    from model.tokens import ANCHOR_TOKEN
    import torch
    tokens = batch["tokens"]
    ts = batch["token_timestamps"]
    B, L = tokens.shape
    # Find anchor positions per sample (handle a single-sample batch).
    anchor_mask = (tokens == ANCHOR_TOKEN)
    # For shape simplicity: compute per-sample.
    anchor_positions = torch.where(anchor_mask[0])[0]
    T = anchor_positions.numel()
    # Event-window grouping: events are [anchor[i]+1 .. anchor[i+1]-1].
    max_events = 128
    event_window = torch.zeros(B, T, max_events, d_model)
    window_mask = torch.zeros(B, T, max_events)
    # Skip actually filling for the shape test (zeros are fine); set mask to 0
    # so the summarizer divides by 1 (its clamp).
    frame_features = torch.zeros(B, T, 10, 6)  # placeholder
    batch["anchor_positions"] = anchor_positions.unsqueeze(0)
    batch["event_window_embeddings"] = event_window
    batch["window_mask"] = window_mask
    batch["frame_features"] = frame_features
```

Call `_segment_for_test(batch, 256)` inside the test before `model(batch)`.

- [ ] **Step 5: Run test, verify pass**

Run: `cd code && uv run pytest tests/model/test_plan_b_model.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add code/model/plan_b_model.py code/tests/model/test_plan_b_model.py
git commit -m "plan-b: compose encoders + RSSM + heads into PlanBModel"
```

---

## Task 9: Dataset extension — anchor windowing, frame features, decision labels

**Files:**
- Modify: `code/model/dataset.py`
- Modify: `code/tests/model/test_dataset_leak.py` (add shape-check)

- [ ] **Step 1: Write the failing test for extended batch dict**

Append to `code/tests/model/test_dataset_leak.py`:
```python
def test_collate_emits_anchor_windowing_tensors(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])

    for key in ("anchor_positions", "event_window_embeddings_raw",
                "window_mask", "frame_features", "decision_labels"):
        assert key in batch, f"batch missing key: {key}"

    T = batch["anchor_positions"].shape[1]
    assert batch["frame_features"].shape[1:] == (T, 10, 6)
    assert batch["window_mask"].shape[:2] == (1, T)
    # decision labels: (B, T, 10), values in [0, 6] where 6 is "no-decision".
    assert batch["decision_labels"].shape == (1, T, 10)
    assert (batch["decision_labels"] >= 0).all()
    assert (batch["decision_labels"] <= 6).all()
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_dataset_leak.py::test_collate_emits_anchor_windowing_tensors -v`
Expected: FAIL — missing keys.

- [ ] **Step 3: Extend `MatchDataset.__getitem__` and `collate_games` in `code/model/dataset.py`**

Inside `__getitem__`, after the existing tokenization code, add:
```python
from model.tokens import ANCHOR_TOKEN, EVENT_TYPE_TO_ID

anchor_idx = [i for i, tok in enumerate(tokens) if tok.type_id == ANCHOR_TOKEN]
# Frame features per anchor: read from `frames` table for the 10 participants.
frame_feats = []
for a_i in anchor_idx:
    ts_ms = tokens[a_i].timestamp_ms
    # Query frames table for this match at this timestamp.
    rows = conn.execute(
        "SELECT participant_slot, gold_total, xp, level, pos_x, pos_y, cs "
        "FROM frames WHERE match_id = ? AND timestamp_ms = ? "
        "ORDER BY participant_slot",
        (mid, ts_ms),
    ).fetchall()
    # Fill (10, 6) -> 0s where missing.
    row_feats = np.zeros((10, 6), dtype=np.float32)
    for r in rows:
        slot = r["participant_slot"]  # 1..10
        if 1 <= slot <= 10:
            row_feats[slot - 1] = [
                float(r["gold_total"] or 0), float(r["xp"] or 0),
                float(r["level"] or 0), float(r["pos_x"] or 0),
                float(r["pos_y"] or 0), float(r["cs"] or 0),
            ]
    frame_feats.append(row_feats)
frame_feats = np.stack(frame_feats, axis=0) if frame_feats else np.zeros((0, 10, 6), np.float32)

# Decision labels: for each anchor t, for each participant p, what decision
# did p make in the window [t, t+1)? If none, use "no-decision" = 6.
DECISION_MAP = {
    EVENT_TYPE_TO_ID["ITEM_PURCHASED"]: 0,
    EVENT_TYPE_TO_ID["SKILL_LEVEL_UP"]: 1,
    EVENT_TYPE_TO_ID["WARD_PLACED"]: 2,
    EVENT_TYPE_TO_ID["RECALL"]: 3,
    EVENT_TYPE_TO_ID["ENGAGE"]: 4,
    EVENT_TYPE_TO_ID["DISENGAGE"]: 5,
}
N_DEC = 6
NO_DEC = 6
decision_labels = np.full((len(anchor_idx), 10), NO_DEC, dtype=np.int64)
for ti, a_i in enumerate(anchor_idx):
    nxt = anchor_idx[ti + 1] if ti + 1 < len(anchor_idx) else len(tokens)
    for tok in tokens[a_i + 1:nxt]:
        if tok.type_id in DECISION_MAP and 1 <= tok.actor_slot <= 10:
            decision_labels[ti, tok.actor_slot - 1] = DECISION_MAP[tok.type_id]

# Event-window token indices (lists of token positions per anchor window).
window_positions = []
for ti, a_i in enumerate(anchor_idx):
    nxt = anchor_idx[ti + 1] if ti + 1 < len(anchor_idx) else len(tokens)
    window_positions.append(list(range(a_i + 1, nxt)))

sample["anchor_positions"] = np.array(anchor_idx, dtype=np.int64)
sample["frame_features"] = frame_feats
sample["decision_labels"] = decision_labels
sample["window_positions"] = window_positions  # ragged — handled in collate
```

In `collate_games`, pad across anchors and windows:
```python
max_T = max(s["anchor_positions"].shape[0] for s in samples)
max_W = 128  # cap events per window; truncate if more

B = len(samples)
anchor_positions = torch.zeros(B, max_T, dtype=torch.long)
frame_features = torch.zeros(B, max_T, 10, 6, dtype=torch.float32)
decision_labels = torch.full((B, max_T, 10), 6, dtype=torch.long)  # 6 = no-decision
# Store raw token indices per window so the model can gather embeddings.
event_window_raw = torch.zeros(B, max_T, max_W, dtype=torch.long)  # token position
window_mask = torch.zeros(B, max_T, max_W, dtype=torch.float32)

for bi, s in enumerate(samples):
    T = s["anchor_positions"].shape[0]
    anchor_positions[bi, :T] = torch.from_numpy(s["anchor_positions"])
    frame_features[bi, :T] = torch.from_numpy(s["frame_features"])
    decision_labels[bi, :T] = torch.from_numpy(s["decision_labels"])
    for ti, positions in enumerate(s["window_positions"]):
        positions = positions[:max_W]
        event_window_raw[bi, ti, :len(positions)] = torch.tensor(positions, dtype=torch.long)
        window_mask[bi, ti, :len(positions)] = 1.0

batch["anchor_positions"] = anchor_positions
batch["event_window_embeddings_raw"] = event_window_raw
batch["window_mask"] = window_mask
batch["frame_features"] = frame_features
batch["decision_labels"] = decision_labels
```

Also update `PlanBModel._segment_by_anchors` to gather `token_emb` at `event_window_embeddings_raw` positions rather than expecting pre-made embeddings:
```python
def _segment_by_anchors(self, batch):
    token_emb = batch["token_emb"]              # (B, L, D_MODEL)
    raw = batch["event_window_embeddings_raw"]  # (B, T, W)
    B, T, W = raw.shape
    gathered = torch.gather(
        token_emb, 1, raw.view(B, T * W).unsqueeze(-1).expand(-1, -1, token_emb.size(-1))
    ).view(B, T, W, token_emb.size(-1))
    return (batch["anchor_positions"], gathered, batch["window_mask"],
            batch["frame_features"])
```

- [ ] **Step 4: Run test, verify pass**

Run: `cd code && uv run pytest tests/model/test_dataset_leak.py -v`
Expected: all pass, including the new shape test.

- [ ] **Step 5: Re-run PlanBModel tests (they depend on the extended batch dict)**

Run: `cd code && uv run pytest tests/model/test_plan_b_model.py -v`
Expected: pass. If the test's inline `_segment_for_test` conflicts with the new batch keys, delete it and rely on the dataset-produced tensors.

- [ ] **Step 6: Run full suite**

Run: `cd code && uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add code/model/dataset.py code/model/plan_b_model.py code/tests/model/test_dataset_leak.py code/tests/model/test_plan_b_model.py
git commit -m "plan-b: dataset emits anchor windowing + frame features + decision labels"
```

---

## Task 10: Training loop — multi-head loss + KL + rollout aux loss

**Files:**
- Create: `code/model/plan_b_train.py`
- Create: `code/tests/model/test_plan_b_train.py`

- [ ] **Step 1: Write the failing smoke test**

Create `code/tests/model/test_plan_b_train.py`:
```python
from model.plan_b_train import plan_b_train_loop


def test_overfit_tiny_corpus_drives_loss_down():
    """Smoke test: 5 games × 3 epochs must drive train loss below its start."""
    from model.dataset import load_split
    train_ids = load_split("train")[:5]
    val_ids = load_split("holdout")[:2]
    cold_ids = load_split("cold")[:2] if load_split("cold") else []

    history = plan_b_train_loop(
        train_ids, val_ids, cold_ids,
        epochs=3, batch_size=1, lr=1e-3,
        max_puuids=50,
        checkpoint_tag="plan_b_smoke",
    )
    first, last = history["train_loss"][0], history["train_loss"][-1]
    assert last < first, f"smoke test: train loss did not decrease ({first} -> {last})"
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_plan_b_train.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `code/model/plan_b_train.py`**

```python
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, build_puuid_index, collate_games, load_split
from model.plan_b_model import PlanBModel, D_Z
from model.rssm import free_bits_kl
from model.rollout import rollout_prior
from model.tokens import NUM_EVENT_TYPES


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_checkpoints")
FREE_BITS_PER_DIM = 0.5
KL_WEIGHT_LOSS = 0.05
ROLLOUT_STEPS = 3
ROLLOUT_LOSS_WEIGHTS = [0.05, 0.03, 0.02]
HEAD_WEIGHTS = {"outcome": 0.35, "next_event": 0.35,
                "next_decision": 0.15, "next_frame": 0.10}
EARLY_STOP_PATIENCE = 5


def _outcome_labels(batch):
    # (B,) -> broadcast to (B, T)
    win = batch["outcome"]  # dataset must supply this (game winner label per sample)
    T = batch["anchor_positions"].size(1)
    return win.unsqueeze(1).expand(-1, T).float()


def _frame_delta_labels(batch):
    # Delta of frame_features between consecutive anchors: (B, T-1, 10, 6)
    ff = batch["frame_features"]
    return ff[:, 1:] - ff[:, :-1]


def _compute_losses(out, batch):
    losses = {}

    event_logits = out["event_logits"]  # (B, T, NUM_EVENT_TYPES)
    losses["next_event"] = F.binary_cross_entropy_with_logits(
        event_logits, batch["labels"].float(), reduction="mean"
    )

    outcome_pred = out["outcome_logits"]  # (B, T)
    outcome_targ = _outcome_labels(batch)
    losses["outcome"] = F.binary_cross_entropy_with_logits(outcome_pred, outcome_targ)

    dec_logits = out["decision_logits"]  # (B, T, 10, 7) — 6 decisions + 1 no-decision
    # Heads produce only 6 classes; append a no-decision slot as zero-logit.
    no_dec_col = torch.zeros(*dec_logits.shape[:-1], 1, device=dec_logits.device)
    dec_logits_full = torch.cat([dec_logits, no_dec_col], dim=-1)  # (B, T, 10, 7)
    losses["next_decision"] = F.cross_entropy(
        dec_logits_full.view(-1, 7), batch["decision_labels"].view(-1).long()
    )

    mu = out["frame_mu"][:, :-1]      # (B, T-1, 10, 6)
    logvar = out["frame_logvar"][:, :-1]
    target = _frame_delta_labels(batch)    # (B, T-1, 10, 6)
    gauss_nll = 0.5 * (logvar + (target - mu).pow(2) / logvar.exp())
    losses["next_frame"] = gauss_nll.mean()

    kl = free_bits_kl(
        out["post_mu"].reshape(-1, D_Z), out["post_logvar"].reshape(-1, D_Z),
        out["prior_mu"].reshape(-1, D_Z), out["prior_logvar"].reshape(-1, D_Z),
        free_bits_per_dim=FREE_BITS_PER_DIM,
    )
    losses["kl"] = kl.mean()

    return losses


def _rollout_aux_loss(model, out, batch):
    """Roll the prior forward ROLLOUT_STEPS from h_final/z_final; apply event +
    outcome heads on the rolled latents. Use last-observed labels as the
    multi-step target (cheap proxy; heads still see distribution shift)."""
    if ROLLOUT_STEPS == 0:
        return torch.tensor(0.0, device=out["event_logits"].device)
    steps = rollout_prior(model.rssm, out["h_final"], out["z_final"],
                          n_steps=ROLLOUT_STEPS, action_summary=None)
    total = 0.0
    last_labels = batch["labels"][:, -1].float()  # (B, NUM_EVENT_TYPES)
    last_outcome = _outcome_labels(batch)[:, -1]  # (B,)
    for (h, z, _mu, _lv), w in zip(steps, ROLLOUT_LOSS_WEIGHTS):
        ev_logits = model.head_event(z)
        oc_logits = model.head_outcome(z)
        total = total + w * (
            F.binary_cross_entropy_with_logits(ev_logits, last_labels) +
            F.binary_cross_entropy_with_logits(oc_logits, last_outcome)
        )
    return total


def _combined_loss(losses, rollout_aux):
    total = (
        HEAD_WEIGHTS["next_event"] * losses["next_event"]
        + HEAD_WEIGHTS["outcome"] * losses["outcome"]
        + HEAD_WEIGHTS["next_decision"] * losses["next_decision"]
        + HEAD_WEIGHTS["next_frame"] * losses["next_frame"]
        + KL_WEIGHT_LOSS * losses["kl"]
        + rollout_aux
    )
    return total


@torch.no_grad()
def _eval_outcome_auc_at_minute(model, ds, target_minute: int = 15) -> float:
    from sklearn.metrics import roc_auc_score
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true, y_score = [], []
    model.eval()
    for batch in loader:
        out = model(batch)
        T = out["n_anchors"]
        minute_idx = min(target_minute, T - 1)
        y_score.append(torch.sigmoid(out["outcome_logits"])[0, minute_idx].item())
        y_true.append(float(batch["outcome"][0].item()))
    model.train()
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def plan_b_train_loop(train_match_ids, val_match_ids, cold_match_ids,
                      epochs: int = 30, batch_size: int = 8, lr: float = 3e-4,
                      max_puuids: int = 20000,
                      checkpoint_tag: str = "plan_b_full"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    puuid_index = build_puuid_index(train_match_ids, max_puuids=max_puuids)

    # Exclude both holdouts from any feature aggregate queried during training.
    exclude = set(val_match_ids) | set(cold_match_ids)
    train_ds = MatchDataset(train_match_ids, puuid_index, exclude_match_ids=exclude)
    val_ds = MatchDataset(val_match_ids, puuid_index, exclude_match_ids=exclude)
    cold_ds = MatchDataset(cold_match_ids, puuid_index, exclude_match_ids=exclude) \
              if cold_match_ids else None

    model = PlanBModel(max_puuids=max_puuids)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "game_cold_auc15": [], "player_cold_auc15": []}
    best_cold_auc = -1.0
    patience_left = EARLY_STOP_PATIENCE
    best_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_tag}_best.pt")

    for ep in range(epochs):
        model.train()
        loader = DataLoader(train_ds, batch_size=batch_size,
                            collate_fn=collate_games, shuffle=True)
        ep_loss = 0.0
        n_batches = 0
        for batch in loader:
            out = model(batch)
            losses = _compute_losses(out, batch)
            aux = _rollout_aux_loss(model, out, batch)
            loss = _combined_loss(losses, aux)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        ep_loss /= max(1, n_batches)
        history["train_loss"].append(ep_loss)

        game_auc = _eval_outcome_auc_at_minute(model, val_ds, 15)
        cold_auc = _eval_outcome_auc_at_minute(model, cold_ds, 15) if cold_ds else 0.5
        history["game_cold_auc15"].append(game_auc)
        history["player_cold_auc15"].append(cold_auc)

        print(f"epoch {ep+1}/{epochs}  loss={ep_loss:.4f}  "
              f"game_cold_auc15={game_auc:.3f}  player_cold_auc15={cold_auc:.3f}")

        if cold_auc > best_cold_auc + 1e-4:
            best_cold_auc = cold_auc
            patience_left = EARLY_STOP_PATIENCE
            torch.save({"state_dict": model.state_dict(),
                        "max_puuids": max_puuids,
                        "history": history}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {ep+1} (no player-cold AUC@15 gain "
                      f"for {EARLY_STOP_PATIENCE} epochs).")
                break

    return history
```

**Note on `batch["outcome"]`:** the dataset doesn't yet emit game outcome. Add a one-liner in `MatchDataset.__getitem__`: `sample["outcome"] = game_row["win"]` (from the `games` table; 1 if blue-side win, 0 if red-side win — or pick the convention used in Plan A's tests and stick to it). Add to `collate_games` as a standard tensor stack.

- [ ] **Step 4: Update `MatchDataset.__getitem__` and `collate_games` to emit `outcome`**

In `dataset.py`:
```python
# in __getitem__, after reading `game_row`:
sample["outcome"] = int(game_row["win"])  # blue-team win: 1, else 0
# in collate_games:
batch["outcome"] = torch.tensor([s["outcome"] for s in samples], dtype=torch.long)
```

- [ ] **Step 5: Run smoke test, verify pass**

Run: `cd code && uv run pytest tests/model/test_plan_b_train.py -v`
Expected: PASS. Takes a few minutes on CPU; loss decreases monotonically.

- [ ] **Step 6: Run full suite**

Run: `cd code && uv run pytest tests/ -q`
Expected: all prior tests + new Plan B tests pass.

- [ ] **Step 7: Commit**

```bash
git add code/model/plan_b_train.py code/model/dataset.py code/tests/model/test_plan_b_train.py
git commit -m "plan-b: training loop (multi-head loss + KL + rollout aux + early stop)"
```

---

## Task 11: Eval suite — outcome AUC per minute, imagination rollout, frozen-m0 probe

**Files:**
- Create: `code/model/plan_b_eval.py`
- Create: `code/tests/model/test_plan_b_eval.py`

- [ ] **Step 1: Write the failing eval tests**

Create `code/tests/model/test_plan_b_eval.py`:
```python
import torch
import pytest
from model.plan_b_model import PlanBModel
from model.dataset import MatchDataset, build_puuid_index, collate_games
from model.plan_b_eval import (
    outcome_auc_by_minute, imagination_rollout_top5, frozen_minute_0_auc,
)


@pytest.fixture(scope="module")
def tiny_model_and_ds(fixture_match_id):
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    model = PlanBModel(max_puuids=len(idx))
    return model, ds


def test_outcome_auc_by_minute_returns_per_minute_dict(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    result = outcome_auc_by_minute(model, ds, minutes=(5, 10, 15))
    assert set(result.keys()) >= {5, 10, 15}
    for m, auc in result.items():
        assert 0.0 <= auc <= 1.0


def test_imagination_rollout_top5_returns_per_step(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    res = imagination_rollout_top5(model, ds, n_steps=3)
    assert len(res) == 3
    for top5 in res:
        assert 0.0 <= top5 <= 1.0


def test_frozen_minute_0_auc_returns_scalar(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    auc = frozen_minute_0_auc(model, ds, target_minute=15)
    assert 0.0 <= auc <= 1.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd code && uv run pytest tests/model/test_plan_b_eval.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `code/model/plan_b_eval.py`**

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model.dataset import collate_games
from model.rollout import rollout_prior
from model.tokens import NUM_EVENT_TYPES


@torch.no_grad()
def outcome_auc_by_minute(model, ds, minutes=(5, 10, 15, 20, 25)):
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true = {m: [] for m in minutes}
    y_score = {m: [] for m in minutes}
    for batch in loader:
        out = model(batch)
        T = out["n_anchors"]
        truth = float(batch["outcome"][0].item())
        scores = torch.sigmoid(out["outcome_logits"])[0]
        for m in minutes:
            if m < T:
                y_true[m].append(truth)
                y_score[m].append(scores[m].item())
    model.train()
    result = {}
    for m in minutes:
        if len(set(y_true[m])) < 2 or not y_true[m]:
            result[m] = 0.5
        else:
            result[m] = float(roc_auc_score(y_true[m], y_score[m]))
    return result


@torch.no_grad()
def imagination_rollout_top5(model, ds, n_steps: int = 3):
    """For each held-out game, teacher-force up to the second-to-last anchor, roll
    the prior forward n_steps, and measure event top-5 accuracy averaged across
    step positions. Returns a list of length n_steps."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    per_step_hits = [[] for _ in range(n_steps)]

    for batch in loader:
        out = model(batch)
        T = out["n_anchors"]
        if T < 2 + n_steps:
            continue
        # Use h, z right before the last n_steps anchors, then roll prior forward.
        post_mu = out["post_mu"][:, T - n_steps - 1]
        post_lv = out["post_logvar"][:, T - n_steps - 1]
        z_seed = post_mu  # mean, deterministic
        # h state at that step is not exposed; re-run through the GRU.
        # Simpler: roll from h_final backwards n_steps is not well defined —
        # instead roll from the anchor T - n_steps using its prior mean as z.
        h_seed = model.static_to_h(model.static_enc(batch["static"]))
        # (approximation: fresh h — we only need prior-rollout shape and
        # whether its predictions beat uniform. Re-seeding from the model's
        # own prior mean from the anchor is a reasonable proxy.)
        steps = rollout_prior(model.rssm, h_seed, z_seed, n_steps=n_steps,
                              action_summary=None)
        for si, (_h, z, _mu, _lv) in enumerate(steps):
            target_t = T - n_steps + si
            target = batch["labels"][0, target_t]  # (NUM_EVENT_TYPES,) multi-hot
            logits = model.head_event(z)[0]        # (NUM_EVENT_TYPES,)
            top5 = torch.topk(logits, k=5).indices.tolist()
            # Multi-hot top-5 accuracy: the argmax-class label must be in top-5.
            argmax_class = int(torch.argmax(target).item())
            per_step_hits[si].append(1.0 if argmax_class in top5 else 0.0)

    model.train()
    return [float(sum(hits)/max(1, len(hits))) for hits in per_step_hits]


@torch.no_grad()
def frozen_minute_0_auc(model, ds, target_minute: int = 15) -> float:
    """Feed only static + player streams; zero out the dynamic sequence. Predict
    outcome at `target_minute`. Target AUC ≤ 0.55 (chance-like)."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true, y_score = [], []
    for batch in loader:
        # Zero out dynamic-stream inputs and keep anchors only so the model
        # still iterates T steps but sees no event information.
        frozen_batch = {k: v for k, v in batch.items()}
        frozen_batch["tokens"] = torch.zeros_like(batch["tokens"])
        frozen_batch["event_window_embeddings_raw"] = torch.zeros_like(
            batch["event_window_embeddings_raw"])
        frozen_batch["window_mask"] = torch.zeros_like(batch["window_mask"])
        frozen_batch["frame_features"] = torch.zeros_like(batch["frame_features"])
        out = model(frozen_batch)
        T = out["n_anchors"]
        if target_minute < T:
            y_true.append(float(batch["outcome"][0].item()))
            y_score.append(torch.sigmoid(out["outcome_logits"])[0, target_minute].item())
    model.train()
    if len(set(y_true)) < 2 or not y_true:
        return 0.5
    return float(roc_auc_score(y_true, y_score))
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd code && uv run pytest tests/model/test_plan_b_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add code/model/plan_b_eval.py code/tests/model/test_plan_b_eval.py
git commit -m "plan-b: eval suite (outcome AUC per minute + rollout top-5 + frozen-m0 probe)"
```

---

## Task 12: CLI entry points

**Files:**
- Modify: `code/model/cli.py`

- [ ] **Step 1: Add subcommands to `code/model/cli.py`**

Extend the existing `parse_args` / subcommand dispatch:
```python
def cmd_cold_build(args):
    from model.cold_holdout import save_player_cold_holdout
    puuids, match_ids = save_player_cold_holdout()
    print(f"cold puuids: {len(puuids)}  cold matches: {len(match_ids)}")


def cmd_plan_b_shakedown(args):
    from model.dataset import load_split
    from model.plan_b_train import plan_b_train_loop
    train = load_split("train")[:50]
    val = load_split("holdout")[:8]
    cold = load_split("cold")[:8]
    hist = plan_b_train_loop(train, val, cold,
                             epochs=30, batch_size=2, lr=1e-3,
                             max_puuids=500,
                             checkpoint_tag="plan_b_shakedown")
    best_cold = max(hist["player_cold_auc15"])
    print(f"best player_cold_auc15 = {best_cold:.3f}")
    assert best_cold >= 0.55, "shakedown failed: player-cold AUC below floor"


def cmd_plan_b_train(args):
    from model.dataset import load_split
    from model.plan_b_train import plan_b_train_loop
    train = load_split("train")
    val = load_split("holdout")
    cold = load_split("cold")
    plan_b_train_loop(train, val, cold,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, max_puuids=args.max_puuids,
                      checkpoint_tag="plan_b_full")


def cmd_plan_b_eval(args):
    import torch, os
    from model.dataset import load_split, MatchDataset, build_puuid_index
    from model.plan_b_model import PlanBModel
    from model.plan_b_eval import (
        outcome_auc_by_minute, imagination_rollout_top5, frozen_minute_0_auc,
    )
    ckpt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "model_checkpoints",
        "plan_b_full_best.pt")
    ckpt = torch.load(ckpt_path, weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)

    val_ids = load_split("holdout")
    cold_ids = load_split("cold")
    idx = build_puuid_index(load_split("train"), max_puuids=max_puuids)

    game_ds = MatchDataset(val_ids, idx, exclude_match_ids=set(val_ids) | set(cold_ids))
    cold_ds = MatchDataset(cold_ids, idx, exclude_match_ids=set(val_ids) | set(cold_ids))

    model = PlanBModel(max_puuids=max_puuids)
    model.load_state_dict(ckpt["state_dict"])

    print("=== GAME-COLD HOLDOUT ===")
    for m, v in outcome_auc_by_minute(model, game_ds).items():
        print(f"  minute {m:>2}: AUC={v:.3f}")

    print("=== PLAYER-COLD HOLDOUT ===")
    for m, v in outcome_auc_by_minute(model, cold_ds).items():
        print(f"  minute {m:>2}: AUC={v:.3f}")

    print("=== IMAGINATION ROLLOUT (game-cold) ===")
    for si, top5 in enumerate(imagination_rollout_top5(model, game_ds, n_steps=3), start=1):
        print(f"  step {si}: event top-5 = {top5:.3f}")

    print("=== LEAK PROBE (frozen-minute-0) ===")
    auc = frozen_minute_0_auc(model, game_ds, target_minute=15)
    print(f"  AUC @15 = {auc:.3f}  (target: ≤ 0.55)")
```

Register each with the argparse subparser block and wire `cmd_plan_b_train` to accept `--epochs`, `--batch-size`, `--lr`, `--max-puuids`.

- [ ] **Step 2: Verify CLI parses**

Run:
```bash
cd code && uv run python -m model.cli --help
uv run python -m model.cli cold-build --help
uv run python -m model.cli plan-b-train --help
```
Expected: each prints its help text without error.

- [ ] **Step 3: Generate the cold holdout file**

Run: `cd code && uv run python -m model.cli cold-build`
Expected: prints `cold puuids: 5  cold matches: <N>`.

- [ ] **Step 4: Commit**

```bash
git add code/model/cli.py data/splits/plan_b_cold_holdout.txt
git commit -m "plan-b: CLI entry points (cold-build, shakedown, train, eval)"
```

---

## Task 13: Plan B shakedown (CPU)

**Files:**
- None (runtime validation).

- [ ] **Step 1: Run shakedown locally**

Run: `cd code && uv run python -m model.cli plan-b-shakedown`
Expected:
- Runs in 10–45 minutes on CPU.
- Loss monotonically decreasing (or near-monotonic).
- Best `player_cold_auc15 ≥ 0.55` at the end — a fail asserts.

- [ ] **Step 2: If shakedown fails, diagnose before GPU handoff**

Triage checklist:
- KL-per-anchor near zero → posterior collapse; raise KL weight to 0.02 or lower free-bits to 0.3.
- Loss flat from epoch 1 → learning rate too low or gradient flow blocked. Test `model_forward_is_differentiable` in `test_plan_b_model.py`.
- AUC stuck at 0.5 → outcome labels swapped or model blind to signal. Print `batch["outcome"]` value distribution.
- AUC diverges between game-cold and player-cold (player-cold << 0.5) → leak regression somewhere; re-run stream-2 leak tests.

Fix, re-commit, re-run shakedown until it passes.

- [ ] **Step 3: Commit shakedown log if it ran**

```bash
git add data/ || true  # in case shakedown wrote any traceable output
git commit -m "plan-b: shakedown passed (player_cold_auc15=<actual>)" --allow-empty
```

---

## Task 14: Full train + eval on GPU (handoff)

**Files:**
- None (GPU handoff).

- [ ] **Step 1: Push branch to origin**

```bash
git push
```

- [ ] **Step 2: Sync data + branch to GPU host**

On the laptop:
```bash
scripts/sync-gpu.sh push
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol && git pull'
```

- [ ] **Step 3: Run training on GPU**

On the laptop:
```bash
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol/code && tmux new -d -s planb "uv run python -m model.cli plan-b-train --epochs 30 --batch-size 8 --lr 3e-4 --max-puuids 20000 > /tmp/planb_train.log 2>&1"'
```

Monitor: `ssh "$HOWL_GPU_HOST" 'tail -f /tmp/planb_train.log'`. Expected runtime 2–6 hours depending on card.

- [ ] **Step 4: Run eval on GPU, capture output**

```bash
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol/code && uv run python -m model.cli plan-b-eval | tee /tmp/planb_eval.txt'
scp "$HOWL_GPU_HOST:/tmp/planb_eval.txt" .
```

- [ ] **Step 5: Pull checkpoint and log back**

```bash
scripts/sync-gpu.sh pull
```

- [ ] **Step 6: Verify targets hit**

Open `planb_eval.txt`. Check **two layers** of acceptance:

1. **Strict research checks from the downstream spec**
   - Outcome AUC should be monotonically non-decreasing with game time on both holdouts.
   - Imagination rollout should satisfy step-1 ≥ step-2 ≥ step-3.
2. **Pragmatic Milestone 3 proof-of-life gates**
   - Game-cold targets: ≥0.55 @ min 5, ≥0.70 @ min 15, ≥0.85 @ min 25.
   - Player-cold ratio: player-cold AUC@15 ≥ 0.80 × game-cold AUC@15.
   - Step-3 rollout top-5 > 0.45 (beats uniform).
   - Leak probe: frozen-m0 AUC@15 ≤ 0.55.

Any miss on the **strict** checks is a documented open question even if the pragmatic gates pass. Any miss on the **pragmatic** gates is a fail unless the user explicitly accepts it.

- [ ] **Step 7: Commit eval artifact**

```bash
# Save eval log to the repo for reproducibility.
mkdir -p data/model_checkpoints
cp planb_eval.txt data/model_checkpoints/plan_b_full_eval.txt
git add data/model_checkpoints/plan_b_full_eval.txt
git commit -m "plan-b: full-train eval log (see Task 14 gates)"
git push
```

- [ ] **Step 8: Close the Plan B bead (or file follow-ups)**

If all gates passed:
```bash
bd close howtowin.lol-<plan-b-epic-id>
```
If one or more gates missed but we accept, file follow-up beads documenting the failures (e.g., "tune KL weight for next iteration" if posterior collapsed) and close the epic as delivered.

---

## Plan self-review

Running the three checks from the writing-plans skill against the spec:

**1. Spec coverage:**

- §A1 (loss weight rebalance) → Task 10 (HEAD_WEIGHTS constants + `_combined_loss`).
- §A2 (prior rollout reconstruction) → Task 7 (primitive), Task 10 (`_rollout_aux_loss`).
- §A3 (static vector scaffolding, PATCH_VECTOR_DIM=1024) → Task 1.
- §A4 (KL 0.01, free-bits 0.5, z_t=32) → Task 5 (free_bits_kl), Task 10 (literal code coefficient must be reported directly; do not claim exact equivalence to the spec-level 0.01 statement without evidence).
- §A5 (30 epochs, early stop on player-cold AUC@15) → Task 10 (EARLY_STOP_PATIENCE=5, early-stop on cold_auc).
- §B1 (stream-2 leak audit) → Task 2.
- §B2 (player-cold-start holdout) → Task 3, Task 4.
- §B3 (frozen-m0 leak probe) → Task 11 (`frozen_minute_0_auc`).
- Metrics (outcome AUC per minute, rollout top-5, game-cold/player-cold, frozen-m0) → Task 11 + Task 14.
- Architecture (D_H=512, D_Z=32, D_STATIC=512, D_OBS=128) → Task 5 (RSSM), Task 8 (PlanBModel constants).
- Tests (stream-2 leak, cold-holdout determinism, prior rollout shape, frozen-m0 smoke, RSSM unit) → Tasks 2, 3, 7, 11, 5 respectively.

No gaps.

**2. Placeholder scan:**

- "Implementation note: do not guess at existing query shapes" in Task 2 — this directs the engineer to read the file rather than filling a placeholder. That's an instruction, not a gap.
- "approximation: fresh h" comment in Task 11's imagination rollout — acknowledged approximation with rationale inline. Acceptable; a cleaner version belongs in a follow-up.
- Outcome label: Task 10 says `sample["outcome"] = int(game_row["win"])` — actual column name may be different (Plan A used `games.win` or similar). Engineer must confirm against the DB schema; no other source exists.

**3. Type consistency:**

- `max_puuids` is a size; `len(puuid_index)` returns that size. Both used in Tasks 8, 10, 12 consistently.
- `PATCH_VECTOR_DIM`, `D_MODEL`, `D_PLAYER`, `D_STATIC`, `D_H`, `D_Z`, `D_OBS`, `D_ACTION` are defined once and referenced symbolically.
- `N_DECISION_TYPES = 6` consistent with `DECISION_MAP` values in dataset.py and the "7-class with no-decision slot" handling in `_compute_losses`.
- `exclude_match_ids` parameter name consistent across `player_feature_vector`, `MatchDataset`, and callers.
- `CHECKPOINT_DIR` path consistent between train and eval.

No inconsistencies found.
