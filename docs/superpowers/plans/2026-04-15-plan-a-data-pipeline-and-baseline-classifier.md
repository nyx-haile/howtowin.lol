# Plan A — Data Pipeline + Baseline Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data tokenization pipeline and a baseline causal-transformer classifier that predicts the next meaningful event in a League of Legends match timeline, satisfying Milestones 1 (pipeline shakedown) and 2 (baseline classifier) of `docs/superpowers/specs/2026-04-15-sequence-model-design.md`.

**Architecture:** New `code/model/` package with clean layering — tokenization (tokens, tokenizer, decisions) → features (patch_params, player_features) → data (dataset) → model (encoders, baseline) → training (train, eval, cli). Consumes the existing SQLite schema (`howtowin.db`, `raw_matches.db`). Uses PyTorch. No RSSM yet — that's Plan B.

**Tech Stack:** Python 3.12, `uv` for package management, PyTorch, SQLite (existing), pytest, Riot Data Dragon (JSON over HTTP).

---

## File Structure

New files under `code/model/`:

- `code/model/__init__.py` — empty marker.
- `code/model/tokens.py` — token vocabulary. Event-type table, participant-slot table, reserved tokens (ANCHOR, PAD, BOS).
- `code/model/decisions.py` — synthetic decision event inference (RECALL, ENGAGE, DISENGAGE) from frames + events.
- `code/model/tokenizer.py` — match_id → ordered token stream (anchors + events + decisions) + per-token attributes.
- `code/model/patch_params.py` — Data Dragon cache + per-game dense patch-parameter vector.
- `code/model/player_features.py` — crafted per-puuid feature vector from historical corpus.
- `code/model/dataset.py` — PyTorch `Dataset` / `DataLoader` wrapping tokenizer + features + labels.
- `code/model/encoders.py` — `StaticContextEncoder`, `PlayerModelEncoder`, `DynamicStreamEmbedder` `nn.Module` classes.
- `code/model/baseline.py` — `CausalTransformerBaseline` model with next-event head.
- `code/model/train.py` — training loop with checkpointing and validation.
- `code/model/eval.py` — top-5 accuracy metric, diagnostics by anchor minute.
- `code/model/cli.py` — entry points (`shakedown`, `train`, `eval`) as a `__main__`.

New tests under `code/tests/model/`:

- `code/tests/model/__init__.py`
- `code/tests/model/conftest.py` — shared fixtures (known good match_id from corpus, small in-memory test DB).
- `code/tests/model/test_tokens.py`
- `code/tests/model/test_decisions.py`
- `code/tests/model/test_tokenizer.py`
- `code/tests/model/test_patch_params.py`
- `code/tests/model/test_player_features.py`
- `code/tests/model/test_dataset.py`
- `code/tests/model/test_encoders.py`
- `code/tests/model/test_baseline.py`
- `code/tests/model/test_eval.py`
- `code/tests/model/test_integration.py` — end-to-end: tokenize → feed model → produce predictions.

Modified files:
- `code/requirements.txt` — add `torch`, `numpy`, `pytest`, `requests` (already present).

Data artifacts:
- `data/dragon_cache/{patch_version}/item.json` — cached Data Dragon item data per patch.
- `data/dragon_cache/{patch_version}/champion.json` — cached Data Dragon champion data per patch.
- `data/model_checkpoints/` — saved model checkpoints.
- `data/splits/plan_a_holdout.txt` — list of match_ids reserved as held-out for M2 eval (fixed split for reproducibility).

---

## Scope and Non-goals for Plan A

**In scope:**
- Milestone 1: pipeline shakedown (overfit tiny corpus to near-zero loss).
- Milestone 2: baseline classifier (top-5 next-meaningful-event accuracy, stretch target ≥ 95%).
- Three-stream input (static, player, dynamic).
- All three input streams wired into a plain causal transformer.
- Next-meaningful-event head only.

**Out of scope (deferred to Plan B onward):**
- RSSM (stochastic latents, prior/posterior, KL loss).
- Frame-feature / outcome / decision decoder heads.
- Retrieval infrastructure.
- Imagination rollouts.
- Lesson generation.

**Acceptance:** Task 14 (M1 shakedown) and Task 15 (M2 held-out eval) both pass.

---

## Task 0: Environment setup

**Files:**
- Modify: `code/requirements.txt`
- Create: `code/model/__init__.py`
- Create: `code/tests/model/__init__.py`

- [ ] **Step 1: Add PyTorch and dev deps to requirements**

Edit `code/requirements.txt` to this exact content:

```
redis
requests
scikit-learn
pandas
torch
numpy
pytest
```

- [ ] **Step 2: Install deps via uv**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv pip install -r requirements.txt`
Expected: installs without errors. Torch download may take 1-2 minutes on first run.

- [ ] **Step 3: Create module markers**

```python
# code/model/__init__.py
# empty
```

```python
# code/tests/model/__init__.py
# empty
```

- [ ] **Step 4: Verify torch imports**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: prints a version like `2.x.x` and either `True` or `False` for CUDA (we don't require CUDA for Plan A; CPU training is slow but sufficient for shakedown and works for the 95% target given corpus size).

- [ ] **Step 5: Commit**

```bash
git add code/requirements.txt code/model/__init__.py code/tests/model/__init__.py
git commit -m "plan-a: scaffold model package and add torch deps"
```

---

## Task 1: Token vocabulary

Define the fixed vocabulary the tokenizer emits. Every token has a type ID and, where applicable, an actor slot (1-10) and a target slot (1-10 or 0 for world/none).

**Files:**
- Create: `code/model/tokens.py`
- Test: `code/tests/model/test_tokens.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_tokens.py
from model.tokens import (
    EVENT_TYPES, EVENT_TYPE_TO_ID, ID_TO_EVENT_TYPE,
    ANCHOR_TOKEN, PAD_TOKEN, BOS_TOKEN,
    NUM_SLOTS, NUM_EVENT_TYPES,
)


def test_reserved_tokens_are_distinct():
    assert len({ANCHOR_TOKEN, PAD_TOKEN, BOS_TOKEN}) == 3


def test_event_types_cover_spec():
    required = {
        "CHAMPION_KILL", "BUILDING_KILL", "ELITE_MONSTER_KILL",
        "ITEM_PURCHASED", "SKILL_LEVEL_UP",
        "WARD_PLACED", "WARD_KILL", "CHAMPION_SPECIAL_KILL",
        "RECALL", "ENGAGE", "DISENGAGE",
    }
    assert required.issubset(set(EVENT_TYPES))


def test_bidirectional_mapping():
    for name in EVENT_TYPES:
        i = EVENT_TYPE_TO_ID[name]
        assert ID_TO_EVENT_TYPE[i] == name


def test_slot_range():
    assert NUM_SLOTS == 11  # slots 0 (world) + 1..10 (participants)


def test_num_event_types_matches_list():
    assert NUM_EVENT_TYPES == len(EVENT_TYPES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.tokens'`.

- [ ] **Step 3: Implement `code/model/tokens.py`**

```python
# code/model/tokens.py
"""Fixed token vocabulary for the world-model input stream.

Meaningful events the spec calls out:
- Outcomes: CHAMPION_KILL, BUILDING_KILL, ELITE_MONSTER_KILL, CHAMPION_SPECIAL_KILL
- Decisions (explicit): ITEM_PURCHASED, SKILL_LEVEL_UP, WARD_PLACED, WARD_KILL
- Decisions (inferred): RECALL, ENGAGE, DISENGAGE

Reserved tokens occupy the first 8 IDs so event-type IDs start at 8.
"""

ANCHOR_TOKEN = 0
PAD_TOKEN = 1
BOS_TOKEN = 2
_RESERVED_COUNT = 8

EVENT_TYPES = [
    "CHAMPION_KILL",
    "BUILDING_KILL",
    "ELITE_MONSTER_KILL",
    "CHAMPION_SPECIAL_KILL",
    "ITEM_PURCHASED",
    "SKILL_LEVEL_UP",
    "WARD_PLACED",
    "WARD_KILL",
    "RECALL",
    "ENGAGE",
    "DISENGAGE",
]

EVENT_TYPE_TO_ID = {name: _RESERVED_COUNT + i for i, name in enumerate(EVENT_TYPES)}
ID_TO_EVENT_TYPE = {i: name for name, i in EVENT_TYPE_TO_ID.items()}

NUM_EVENT_TYPES = len(EVENT_TYPES)
NUM_SLOTS = 11  # slot 0 = world/none; slots 1..10 = participants
VOCAB_SIZE = _RESERVED_COUNT + NUM_EVENT_TYPES
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_tokens.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/model/tokens.py code/tests/model/test_tokens.py
git commit -m "plan-a: define event-token vocabulary"
```

---

## Task 2: Test fixtures

Establish shared pytest fixtures so every downstream test uses the same known-good match from the real corpus.

**Files:**
- Create: `code/tests/model/conftest.py`

- [ ] **Step 1: Identify a known-good match_id**

Run this one-liner to pick the match_id of the oldest game (stable, unlikely to be evicted):

```bash
cd /home/lunaris/build/howtowin.lol && sqlite3 data/howtowin.db "SELECT match_id FROM games ORDER BY created_at ASC LIMIT 1"
```

Record the returned match_id — call it `FIXTURE_MATCH_ID`. If the corpus is empty, run `cd code && uv run python test/direct_parse.py` first to ingest a handful of games, then retry.

- [ ] **Step 2: Write conftest with fixtures**

Replace `{{FIXTURE_MATCH_ID}}` with the recorded value.

```python
# code/tests/model/conftest.py
import os
import sys
import pytest

# Make `import model.xxx` work when running pytest from code/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FIXTURE_MATCH_ID = "{{FIXTURE_MATCH_ID}}"


@pytest.fixture(scope="session")
def fixture_match_id():
    return FIXTURE_MATCH_ID


@pytest.fixture(scope="session")
def all_match_ids():
    """All parsed match_ids in the corpus. Used to exercise dataset loading."""
    from db import get_conn
    conn = get_conn()
    rows = conn.execute("SELECT match_id FROM games ORDER BY created_at ASC").fetchall()
    conn.close()
    return [r["match_id"] for r in rows]
```

- [ ] **Step 3: Verify fixture resolves**

Write a quick ad-hoc test and run it:

```python
# code/tests/model/test_fixtures_smoke.py  (temporary, delete in step 5)
def test_fixture(fixture_match_id, all_match_ids):
    assert fixture_match_id.startswith("NA1_") or fixture_match_id.startswith("EUW1_") or len(fixture_match_id) > 0
    assert fixture_match_id in all_match_ids
```

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_fixtures_smoke.py -v`
Expected: PASS.

- [ ] **Step 4: Delete smoke test**

```bash
rm code/tests/model/test_fixtures_smoke.py
```

- [ ] **Step 5: Commit**

```bash
git add code/tests/model/conftest.py
git commit -m "plan-a: add shared test fixtures anchored on corpus match"
```

---

## Task 3: Recall inference

Detect per-participant RECALL events by position-jump to the team fountain. Fountains on Summoner's Rift are at roughly `(~400, ~400)` for blue team and `(~14300, ~14300)` for red team (map size is 15000x15000).

**Files:**
- Create: `code/model/decisions.py`
- Test: `code/tests/model/test_decisions.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_decisions.py
from model.decisions import infer_recalls, FOUNTAIN_RADIUS, BLUE_FOUNTAIN, RED_FOUNTAIN


def test_recall_detected_on_fountain_jump():
    # Frame t=60s: participant 1 (blue team) mid-lane-ish
    # Frame t=120s: same participant at fountain
    frames_by_ts = {
        60000: [
            {"slot": 1, "team_id": 100, "pos_x": 7000, "pos_y": 7000},
        ],
        120000: [
            {"slot": 1, "team_id": 100, "pos_x": BLUE_FOUNTAIN[0] + 100, "pos_y": BLUE_FOUNTAIN[1] + 100},
        ],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 1
    assert recalls[0]["slot"] == 1
    assert recalls[0]["timestamp_ms"] == 120000


def test_no_recall_when_already_at_fountain():
    frames_by_ts = {
        60000: [
            {"slot": 1, "team_id": 100, "pos_x": BLUE_FOUNTAIN[0], "pos_y": BLUE_FOUNTAIN[1]},
        ],
        120000: [
            {"slot": 1, "team_id": 100, "pos_x": BLUE_FOUNTAIN[0] + 50, "pos_y": BLUE_FOUNTAIN[1] + 50},
        ],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 0


def test_red_team_uses_red_fountain():
    frames_by_ts = {
        60000: [{"slot": 6, "team_id": 200, "pos_x": 7000, "pos_y": 7000}],
        120000: [{"slot": 6, "team_id": 200, "pos_x": RED_FOUNTAIN[0] - 100, "pos_y": RED_FOUNTAIN[1] - 100}],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 1
    assert recalls[0]["slot"] == 6


def test_red_at_blue_fountain_is_not_recall():
    # Red team participant near blue fountain = invading, not recalling
    frames_by_ts = {
        60000: [{"slot": 6, "team_id": 200, "pos_x": 7000, "pos_y": 7000}],
        120000: [{"slot": 6, "team_id": 200, "pos_x": BLUE_FOUNTAIN[0] + 50, "pos_y": BLUE_FOUNTAIN[1] + 50}],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 0


def test_fountain_radius_positive():
    assert FOUNTAIN_RADIUS > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_decisions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.decisions'`.

- [ ] **Step 3: Implement recall inference**

```python
# code/model/decisions.py
"""Infer decision events (RECALL, ENGAGE, DISENGAGE) from frames + events.

The Riot timeline does not emit explicit "recall" events — players just
teleport home. We infer it by detecting a position jump into the team's
fountain region between consecutive frames.

ENGAGE/DISENGAGE are inferred from team-position clustering around
CHAMPION_KILL events (Task 4).
"""

BLUE_FOUNTAIN = (400, 400)
RED_FOUNTAIN = (14300, 14300)
FOUNTAIN_RADIUS = 1500  # generous; covers spawn pad + inhibitor area


def _in_fountain(pos_x, pos_y, team_id):
    fx, fy = BLUE_FOUNTAIN if team_id == 100 else RED_FOUNTAIN
    dx = pos_x - fx
    dy = pos_y - fy
    return (dx * dx + dy * dy) <= (FOUNTAIN_RADIUS * FOUNTAIN_RADIUS)


def infer_recalls(frames_by_ts):
    """Return list of {slot, timestamp_ms} for recalls.

    frames_by_ts: dict mapping timestamp_ms -> list of
                  {slot, team_id, pos_x, pos_y}.
    A recall fires when a participant moves INTO their team's fountain
    between timestamp t-1 and t.
    """
    sorted_ts = sorted(frames_by_ts.keys())
    prev_positions = {}  # slot -> (pos_x, pos_y, team_id, in_fountain)
    recalls = []

    for ts in sorted_ts:
        for f in frames_by_ts[ts]:
            slot = f["slot"]
            tid = f["team_id"]
            px, py = f["pos_x"], f["pos_y"]
            now_in = _in_fountain(px, py, tid)
            prev = prev_positions.get(slot)
            if prev is not None:
                _, _, _, prev_in = prev
                if now_in and not prev_in:
                    recalls.append({"slot": slot, "timestamp_ms": ts, "team_id": tid})
            prev_positions[slot] = (px, py, tid, now_in)

    return recalls
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_decisions.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/model/decisions.py code/tests/model/test_decisions.py
git commit -m "plan-a: infer RECALL events from fountain-region position jumps"
```

---

## Task 4: Engage/disengage inference

Coarse team-level engage/disengage: for each CHAMPION_KILL, look at participant positions 15 seconds before the kill. If ≥3 attackers-team members and ≥2 victim-team members were within a fighting radius, flag an ENGAGE by the attacker team. If the attacker team has *fewer* members present than the victim team had, flag a DISENGAGE attempt by the victim team that failed (defensive engage).

**Files:**
- Modify: `code/model/decisions.py`
- Modify: `code/tests/model/test_decisions.py`

- [ ] **Step 1: Write the failing test (appended)**

Append to `code/tests/model/test_decisions.py`:

```python
from model.decisions import infer_engages, FIGHT_RADIUS, PRE_FIGHT_WINDOW_MS


def test_engage_detected_on_clustered_kill():
    # CHAMPION_KILL at t=300000 with killer slot 1, victim slot 6.
    # At t=285000 (15s pre-fight), 3 blue members and 2 red members clustered.
    frames_by_ts = {
        285000: [
            {"slot": 1, "team_id": 100, "pos_x": 7500, "pos_y": 7500},
            {"slot": 2, "team_id": 100, "pos_x": 7600, "pos_y": 7500},
            {"slot": 3, "team_id": 100, "pos_x": 7500, "pos_y": 7600},
            {"slot": 6, "team_id": 200, "pos_x": 7700, "pos_y": 7700},
            {"slot": 7, "team_id": 200, "pos_x": 7800, "pos_y": 7700},
        ],
    }
    kill_events = [
        {"timestamp_ms": 300000, "killer_id": 1, "victim_id": 6,
         "position_x": 7700, "position_y": 7700},
    ]
    tags = infer_engages(kill_events, frames_by_ts)
    engages = [t for t in tags if t["event_type"] == "ENGAGE"]
    assert len(engages) == 1
    assert engages[0]["team_id"] == 100


def test_no_engage_when_positions_far():
    # Only one blue nearby — not an engage, just a pick.
    frames_by_ts = {
        285000: [
            {"slot": 1, "team_id": 100, "pos_x": 7500, "pos_y": 7500},
            {"slot": 6, "team_id": 200, "pos_x": 7700, "pos_y": 7700},
        ],
    }
    kill_events = [
        {"timestamp_ms": 300000, "killer_id": 1, "victim_id": 6,
         "position_x": 7700, "position_y": 7700},
    ]
    tags = infer_engages(kill_events, frames_by_ts)
    assert len(tags) == 0


def test_fight_radius_and_window_positive():
    assert FIGHT_RADIUS > 0
    assert PRE_FIGHT_WINDOW_MS > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_decisions.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'infer_engages'`.

- [ ] **Step 3: Implement engage/disengage inference**

Append to `code/model/decisions.py`:

```python
FIGHT_RADIUS = 2500
PRE_FIGHT_WINDOW_MS = 15000
MIN_ATTACKERS = 3
MIN_DEFENDERS = 2


def _nearest_frame_ts(frames_by_ts, target_ts):
    """Return the frame timestamp closest to target_ts but not after it."""
    candidates = [ts for ts in frames_by_ts.keys() if ts <= target_ts]
    if not candidates:
        return None
    return max(candidates)


def _count_nearby(frames, cx, cy, team_id):
    n = 0
    for f in frames:
        if f["team_id"] != team_id:
            continue
        dx = f["pos_x"] - cx
        dy = f["pos_y"] - cy
        if dx * dx + dy * dy <= FIGHT_RADIUS * FIGHT_RADIUS:
            n += 1
    return n


def infer_engages(kill_events, frames_by_ts):
    """Return list of {event_type: ENGAGE|DISENGAGE, team_id, timestamp_ms}.

    kill_events: list of {timestamp_ms, killer_id, victim_id, position_x, position_y}.
    frames_by_ts: same shape as infer_recalls.

    Heuristic: for each kill, look at frames ~PRE_FIGHT_WINDOW_MS before the kill.
    Count attackers and defenders within FIGHT_RADIUS of the kill position.
    If the attacker count >= MIN_ATTACKERS and defender count >= MIN_DEFENDERS,
    emit an ENGAGE tagged to the attacker team. If defender count exceeds
    attacker count at that moment, also emit a DISENGAGE on the defender team
    (the defending team tried to contest and got killed).
    """
    seen_team_ts = set()  # dedupe: one engage per team per kill-timestamp
    tags = []

    for ev in kill_events:
        kts = ev["timestamp_ms"]
        pre_ts = _nearest_frame_ts(frames_by_ts, kts - PRE_FIGHT_WINDOW_MS)
        if pre_ts is None:
            continue
        frames = frames_by_ts[pre_ts]

        killer_slot = ev.get("killer_id", 0)
        victim_slot = ev.get("victim_id", 0)
        if not (1 <= killer_slot <= 10) or not (1 <= victim_slot <= 10):
            continue

        killer_team = 100 if killer_slot <= 5 else 200
        victim_team = 100 if victim_slot <= 5 else 200
        if killer_team == victim_team:
            continue  # jungle monster kills etc.

        cx, cy = ev["position_x"], ev["position_y"]
        attackers = _count_nearby(frames, cx, cy, killer_team)
        defenders = _count_nearby(frames, cx, cy, victim_team)

        if attackers >= MIN_ATTACKERS and defenders >= MIN_DEFENDERS:
            key = ("ENGAGE", killer_team, kts)
            if key not in seen_team_ts:
                tags.append({"event_type": "ENGAGE", "team_id": killer_team, "timestamp_ms": kts})
                seen_team_ts.add(key)
            if defenders > attackers:
                key2 = ("DISENGAGE", victim_team, kts)
                if key2 not in seen_team_ts:
                    tags.append({"event_type": "DISENGAGE", "team_id": victim_team, "timestamp_ms": kts})
                    seen_team_ts.add(key2)

    return tags
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_decisions.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/model/decisions.py code/tests/model/test_decisions.py
git commit -m "plan-a: infer team-level ENGAGE/DISENGAGE from position clustering at kills"
```

---

## Task 5: Tokenizer

Given a `match_id`, read the parsed tables + raw blob and produce an ordered token stream: anchor tokens at each minute boundary, event tokens at their `timestamp_ms`, inferred decision tokens interleaved.

**Files:**
- Create: `code/model/tokenizer.py`
- Test: `code/tests/model/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_tokenizer.py
from model.tokenizer import tokenize_match, Token
from model.tokens import ANCHOR_TOKEN, EVENT_TYPE_TO_ID


def test_tokenize_returns_non_empty_stream(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    assert len(stream) > 0


def test_stream_starts_with_anchor_at_t0(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    first = stream[0]
    assert first.type_id == ANCHOR_TOKEN
    assert first.timestamp_ms == 0


def test_stream_sorted_by_timestamp(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    timestamps = [t.timestamp_ms for t in stream]
    assert timestamps == sorted(timestamps)


def test_anchor_interval_is_60s(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    anchor_ts = [t.timestamp_ms for t in stream if t.type_id == ANCHOR_TOKEN]
    assert len(anchor_ts) >= 2
    # Consecutive anchor gaps should be 60000ms (Riot timeline cadence).
    # Riot emits the final frame at match end, not on the minute boundary, so
    # the last gap may be short; exclude it from the tolerance check.
    gaps = [anchor_ts[i + 1] - anchor_ts[i] for i in range(len(anchor_ts) - 1)]
    assert all(abs(g - 60000) <= 1000 for g in gaps[:-1])


def test_event_tokens_reference_known_slots(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    kill_id = EVENT_TYPE_TO_ID["CHAMPION_KILL"]
    kills = [t for t in stream if t.type_id == kill_id]
    for k in kills:
        assert 1 <= k.actor_slot <= 10
        assert 0 <= k.target_slot <= 10


def test_token_dataclass_fields():
    t = Token(type_id=0, actor_slot=0, target_slot=0, timestamp_ms=0)
    assert hasattr(t, "type_id") and hasattr(t, "actor_slot")
    assert hasattr(t, "target_slot") and hasattr(t, "timestamp_ms")


def test_recall_tokens_inferred(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    recall_id = EVENT_TYPE_TO_ID["RECALL"]
    recalls = [t for t in stream if t.type_id == recall_id]
    # A 30+ minute game always has recalls.
    assert len(recalls) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.tokenizer'`.

- [ ] **Step 3: Implement the tokenizer**

```python
# code/model/tokenizer.py
"""match_id -> ordered token stream.

Stream layout:
  - One ANCHOR token per minute boundary (t=0, 60000, 120000, ...).
  - One event token per meaningful event at its timestamp_ms.
  - Inferred RECALL / ENGAGE / DISENGAGE tokens from decisions.py.

The stream is globally sorted by timestamp_ms. Anchors preceding events at
the same timestamp come first.
"""
from dataclasses import dataclass
from collections import defaultdict

from db import get_conn
from model.tokens import ANCHOR_TOKEN, EVENT_TYPE_TO_ID
from model.decisions import infer_recalls, infer_engages


MEANINGFUL_EVENT_TYPES = {
    "CHAMPION_KILL", "BUILDING_KILL", "ELITE_MONSTER_KILL", "CHAMPION_SPECIAL_KILL",
    "ITEM_PURCHASED", "SKILL_LEVEL_UP", "WARD_PLACED", "WARD_KILL",
}


@dataclass
class Token:
    type_id: int
    actor_slot: int  # 0 = world/none, 1-10 = participant
    target_slot: int  # 0 = none, 1-10 = participant
    timestamp_ms: int


def _load_events(conn, match_id):
    rows = conn.execute(
        """SELECT timestamp_ms, event_type, participant_id, killer_id, victim_id,
                  position_x, position_y
           FROM events WHERE match_id = ? ORDER BY timestamp_ms""",
        (match_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def _load_frames(conn, match_id):
    rows = conn.execute(
        """SELECT timestamp_ms, participant_slot, team_id, pos_x, pos_y
           FROM frames WHERE match_id = ? ORDER BY timestamp_ms, participant_slot""",
        (match_id,)
    ).fetchall()
    by_ts = defaultdict(list)
    for r in rows:
        by_ts[r["timestamp_ms"]].append({
            "slot": r["participant_slot"],
            "team_id": r["team_id"],
            "pos_x": r["pos_x"],
            "pos_y": r["pos_y"],
        })
    return by_ts


def _actor_of(ev):
    etype = ev["event_type"]
    if etype == "CHAMPION_KILL":
        return ev["killer_id"] or 0
    if etype == "BUILDING_KILL" or etype == "ELITE_MONSTER_KILL":
        return ev["killer_id"] or 0
    if etype == "CHAMPION_SPECIAL_KILL":
        return ev["killer_id"] or 0
    if etype == "ITEM_PURCHASED" or etype == "SKILL_LEVEL_UP":
        return ev["participant_id"] or 0
    if etype == "WARD_PLACED" or etype == "WARD_KILL":
        return ev["participant_id"] or 0
    return 0


def _target_of(ev):
    if ev["event_type"] == "CHAMPION_KILL":
        return ev["victim_id"] or 0
    return 0


def tokenize_match(match_id):
    conn = get_conn()
    try:
        events = _load_events(conn, match_id)
        frames_by_ts = _load_frames(conn, match_id)
    finally:
        conn.close()

    tokens = []

    # Anchor tokens, one per frame timestamp.
    for ts in sorted(frames_by_ts.keys()):
        tokens.append(Token(ANCHOR_TOKEN, 0, 0, ts))

    # Meaningful events from timeline.
    for ev in events:
        etype = ev["event_type"]
        if etype not in MEANINGFUL_EVENT_TYPES:
            continue
        type_id = EVENT_TYPE_TO_ID[etype]
        actor = _actor_of(ev)
        target = _target_of(ev)
        if not (1 <= actor <= 10):
            continue  # world-owned events (Rift Herald spawn etc.) skipped for now
        tokens.append(Token(type_id, actor, target, ev["timestamp_ms"]))

    # Inferred decisions.
    for r in infer_recalls(frames_by_ts):
        tokens.append(Token(EVENT_TYPE_TO_ID["RECALL"], r["slot"], 0, r["timestamp_ms"]))

    kill_events = [ev for ev in events if ev["event_type"] == "CHAMPION_KILL"
                   and ev["position_x"] is not None]
    for tag in infer_engages(kill_events, frames_by_ts):
        # Team-level: actor_slot encodes team — 100 -> slot 0 with team bit.
        # Simpler: record first participant slot of that team (1 for blue, 6 for red).
        anchor_slot = 1 if tag["team_id"] == 100 else 6
        tokens.append(Token(
            EVENT_TYPE_TO_ID[tag["event_type"]],
            anchor_slot, 0, tag["timestamp_ms"],
        ))

    # Sort: anchors before events at the same ts (ANCHOR_TOKEN == 0, min id).
    tokens.sort(key=lambda t: (t.timestamp_ms, t.type_id))
    return tokens
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_tokenizer.py -v`
Expected: all 7 tests PASS. If `test_recall_tokens_inferred` fails, the fixture match may be too short — pick a longer match as fixture.

- [ ] **Step 5: Commit**

```bash
git add code/model/tokenizer.py code/tests/model/test_tokenizer.py
git commit -m "plan-a: tokenize match into anchor + event + decision stream"
```

---

## Task 6: Patch parameter extraction

Fetch/cache Data Dragon JSON per patch, produce a dense per-game patch-parameter vector.

**Files:**
- Create: `code/model/patch_params.py`
- Test: `code/tests/model/test_patch_params.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_patch_params.py
import numpy as np
from model.patch_params import (
    fetch_dragon_version, load_items, load_champions,
    patch_vector_for_match, PATCH_VECTOR_DIM,
)


def test_dragon_version_format():
    v = fetch_dragon_version("14.14.1")
    assert v == "14.14.1"


def test_load_items_returns_dict():
    items = load_items("14.14.1")
    # At least a few well-known items in 14.x.
    assert any("BF Sword" in i.get("name", "") for i in items.values()) or \
           any("B. F. Sword" in i.get("name", "") for i in items.values())


def test_load_champions_returns_dict():
    champs = load_champions("14.14.1")
    # 160+ champs by late season 14.
    assert len(champs) > 150


def test_patch_vector_shape(fixture_match_id):
    vec = patch_vector_for_match(fixture_match_id)
    assert vec.shape == (PATCH_VECTOR_DIM,)
    assert vec.dtype == np.float32


def test_patch_vector_is_finite(fixture_match_id):
    vec = patch_vector_for_match(fixture_match_id)
    assert np.all(np.isfinite(vec))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_patch_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.patch_params'`.

- [ ] **Step 3: Implement patch_params**

```python
# code/model/patch_params.py
"""Data Dragon cache + per-game patch-parameter vector.

Pulls item.json and champion.json from Community Dragon CDN, caches locally
under data/dragon_cache/<patch>/. The per-game vector is a fixed-length
dense summary: champion stats (base stats + Q/W/E/R info rank 1-5 for
picks present) + aggregated item stats present in the game. The exact
dimension is controlled by PATCH_VECTOR_DIM and the fields enumerated
below; keep stable across Plan A so the encoder layer has a fixed input.
"""
import json
import os
import requests
import numpy as np

from db import get_conn

DRAGON_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dragon_cache')
DDRAGON_BASE = "https://ddragon.leagueoflegends.com/cdn"

# Dimension breakdown (total 256):
#   10 picks * 10 stat fields = 100  (base stats per champion present)
#   Aggregated item-stat totals across all participants' final builds = 20
#   Global constants (drake HP, baron gold, plate gold) = 10
#   Version one-hot proxy (major, minor, point) = 3
#   Reserved zeros for Plan A = 123
PATCH_VECTOR_DIM = 256

CHAMP_STAT_FIELDS = ["hp", "hpperlevel", "mp", "mpperlevel", "armor",
                     "armorperlevel", "attackdamage", "attackdamageperlevel",
                     "attackspeedoffset", "attackspeed"]

ITEM_STAT_FIELDS = ["FlatPhysicalDamageMod", "FlatMagicDamageMod",
                    "FlatArmorMod", "FlatSpellBlockMod", "FlatHPPoolMod",
                    "FlatMPPoolMod", "PercentAttackSpeedMod",
                    "FlatCritChanceMod", "FlatMovementSpeedMod", "FlatHPRegenMod",
                    "PercentLifeStealMod", "FlatBlockMod",
                    "FlatEnergyPoolMod", "FlatMagicPenetrationMod",
                    "FlatPhysicalDamageReduction", "FlatArmorPenetrationMod",
                    "PercentMovementSpeedMod", "PercentArmorMod",
                    "PercentMagicPenetrationMod", "PercentCritChanceMod"]


def fetch_dragon_version(patch):
    """Given a match's gameVersion like '14.14.580.1234', return a Data Dragon
    patch id like '14.14.1'. For Plan A, default to '14.14.1' when match
    patches don't resolve cleanly."""
    if not patch:
        return "14.14.1"
    parts = patch.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.1"
    return "14.14.1"


def _cache_path(version, filename):
    return os.path.join(DRAGON_CACHE_DIR, version, filename)


def _download(version, filename):
    url = f"{DDRAGON_BASE}/{version}/data/en_US/{filename}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path = _cache_path(version, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(resp.text)
    return resp.json()


def _load_json(version, filename):
    path = _cache_path(version, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return _download(version, filename)


def load_items(version):
    return _load_json(version, "item.json")["data"]


def load_champions(version):
    raw = _load_json(version, "champion.json")["data"]
    # champion.json summaries are keyed by champ name; stats are nested.
    return {cname: cdata for cname, cdata in raw.items()}


def _champ_name_by_key(champs, key):
    """Data Dragon uses champ names as keys; Riot match API uses numeric
    championId ('key' field). Build reverse index."""
    for name, data in champs.items():
        if str(data.get("key")) == str(key):
            return name
    return None


def patch_vector_for_match(match_id):
    conn = get_conn()
    try:
        game = conn.execute(
            "SELECT patch FROM games WHERE match_id = ?", (match_id,)
        ).fetchone()
        raw_picks_rows = conn.execute(
            "SELECT match_id FROM games WHERE match_id = ?", (match_id,)
        ).fetchall()
    finally:
        conn.close()

    version = fetch_dragon_version(game["patch"] if game else "")
    champs = load_champions(version)
    items = load_items(version)

    vec = np.zeros(PATCH_VECTOR_DIM, dtype=np.float32)

    # Picks: read from raw match via raw_db.
    from raw_db import get_raw_match
    match, _tl = get_raw_match(match_id)
    offset = 0
    if match:
        for i, p in enumerate(match["info"]["participants"][:10]):
            champ_key = p.get("championId")
            champ_name = _champ_name_by_key(champs, champ_key)
            if champ_name and champ_name in champs:
                stats = champs[champ_name].get("stats", {})
                for j, fname in enumerate(CHAMP_STAT_FIELDS):
                    vec[offset + i * len(CHAMP_STAT_FIELDS) + j] = float(stats.get(fname, 0.0))
    offset = 10 * len(CHAMP_STAT_FIELDS)  # = 100

    # Aggregated item stats from final builds.
    if match:
        item_totals = {f: 0.0 for f in ITEM_STAT_FIELDS}
        for p in match["info"]["participants"][:10]:
            for slot in range(7):
                iid = p.get(f"item{slot}", 0)
                if iid and str(iid) in items:
                    idata = items[str(iid)]
                    for fname in ITEM_STAT_FIELDS:
                        item_totals[fname] += float(idata.get("stats", {}).get(fname, 0.0))
        for j, fname in enumerate(ITEM_STAT_FIELDS):
            vec[offset + j] = item_totals[fname]
    offset += len(ITEM_STAT_FIELDS)  # = 120

    # Version one-hot proxy.
    parts = version.split(".")
    vec[offset] = float(parts[0]) if parts and parts[0].isdigit() else 0.0
    vec[offset + 1] = float(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0.0
    vec[offset + 2] = float(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0.0

    return vec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_patch_params.py -v`
Expected: all 5 tests PASS. First run downloads Data Dragon JSON (~20-30 MB); later runs use the local cache.

- [ ] **Step 5: Commit**

```bash
git add code/model/patch_params.py code/tests/model/test_patch_params.py
git commit -m "plan-a: Data Dragon cache + per-game patch-parameter vector"
```

---

## Task 7: Player features

Per-puuid crafted feature vector, derived from the existing `players` and `frames`/`events` tables.

**Files:**
- Create: `code/model/player_features.py`
- Test: `code/tests/model/test_player_features.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_player_features.py
import numpy as np
from model.player_features import (
    player_feature_vector, PLAYER_FEATURE_DIM, RANK_TIER_ORDER,
)


def test_feature_shape_for_puuid_in_corpus(all_match_ids, fixture_match_id):
    from raw_db import get_raw_match
    match, _ = get_raw_match(fixture_match_id)
    puuid = match["info"]["participants"][0]["puuid"]
    vec = player_feature_vector(puuid)
    assert vec.shape == (PLAYER_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_unknown_puuid_returns_zero_vector():
    vec = player_feature_vector("nonexistent-puuid-xxx")
    assert vec.shape == (PLAYER_FEATURE_DIM,)
    assert np.allclose(vec, 0.0)


def test_rank_tier_order_monotonic():
    # CHALLENGER > GRANDMASTER > MASTER > DIAMOND > ...
    assert RANK_TIER_ORDER["CHALLENGER"] > RANK_TIER_ORDER["GRANDMASTER"]
    assert RANK_TIER_ORDER["GRANDMASTER"] > RANK_TIER_ORDER["MASTER"]
    assert RANK_TIER_ORDER["DIAMOND"] > RANK_TIER_ORDER["EMERALD"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_player_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.player_features'`.

- [ ] **Step 3: Implement player_features**

```python
# code/model/player_features.py
"""Per-puuid crafted feature vector.

Fields (total 16):
  0: rank_tier_ordinal     (0 for unranked; higher = higher tier)
  1: lp_normalized         (lp / 1000, capped at 2.0 for Masters+)
  2: games_in_corpus       (log1p)
  3: winrate_in_corpus     (wins / games, 0 if games=0)
  4: avg_kda               ((kills+assists)/max(deaths,1))
  5: avg_cs_at_10          (total CS at minute 10, averaged across corpus)
  6: avg_gold_at_10
  7: avg_damage_to_champs
  8: avg_wards_placed
  9: avg_wards_killed
 10: main_role_top
 11: main_role_jungle
 12: main_role_mid
 13: main_role_bot
 14: main_role_support
 15: reserved
"""
import numpy as np

from db import get_conn

PLAYER_FEATURE_DIM = 16

RANK_TIER_ORDER = {
    "IRON": 1, "BRONZE": 2, "SILVER": 3, "GOLD": 4, "PLATINUM": 5,
    "EMERALD": 6, "DIAMOND": 7, "MASTER": 8, "GRANDMASTER": 9, "CHALLENGER": 10,
}

ROLE_INDEX = {"TOP": 10, "JGL": 11, "MID": 12, "BOT": 13, "SUP": 14}


def player_feature_vector(puuid):
    vec = np.zeros(PLAYER_FEATURE_DIM, dtype=np.float32)
    conn = get_conn()
    try:
        prow = conn.execute(
            "SELECT rank_tier, lp FROM players WHERE puuid = ?", (puuid,)
        ).fetchone()
        if prow is None:
            return vec

        if prow["rank_tier"]:
            vec[0] = float(RANK_TIER_ORDER.get(prow["rank_tier"], 0))
        if prow["lp"] is not None:
            vec[1] = min(float(prow["lp"]) / 1000.0, 2.0)

        # Games in corpus for this puuid.
        games = conn.execute(
            """SELECT f.match_id, f.team_id, g.winning_team
               FROM frames f JOIN games g USING(match_id)
               WHERE f.puuid = ? GROUP BY f.match_id""",
            (puuid,)
        ).fetchall()
        n = len(games)
        vec[2] = float(np.log1p(n))
        if n > 0:
            wins = sum(1 for g in games if g["team_id"] == g["winning_team"])
            vec[3] = float(wins) / float(n)

        # Per-game aggregates at minute 10 (timestamp ~600000ms).
        rows = conn.execute(
            """SELECT cs, total_gold, kills, deaths, assists, ward_count
               FROM frames WHERE puuid = ? AND timestamp_ms BETWEEN 540000 AND 660000""",
            (puuid,)
        ).fetchall()
        if rows:
            cs = [r["cs"] for r in rows]
            gold = [r["total_gold"] for r in rows]
            kda = [(r["kills"] + r["assists"]) / max(r["deaths"], 1) for r in rows]
            vec[4] = float(np.mean(kda))
            vec[5] = float(np.mean(cs))
            vec[6] = float(np.mean(gold))
            vec[8] = float(np.mean([r["ward_count"] for r in rows]))

        # Damage placeholder (not populated in frames table; leave 0).
        # Damage will be added when the schema carries it; zero is a safe default.

        # Wards killed approximated by ward_count delta; skipping for Plan A.

        # Role frequency.
        roles = conn.execute(
            "SELECT role, COUNT(*) c FROM frames WHERE puuid = ? GROUP BY role",
            (puuid,)
        ).fetchall()
        total = sum(r["c"] for r in roles) or 1
        for r in roles:
            idx = ROLE_INDEX.get(r["role"])
            if idx is not None:
                vec[idx] = float(r["c"]) / float(total)
    finally:
        conn.close()
    return vec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_player_features.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/model/player_features.py code/tests/model/test_player_features.py
git commit -m "plan-a: per-puuid crafted feature vector from corpus history"
```

---

## Task 8: Dataset + holdout split

PyTorch `Dataset` that yields one game at a time: static context + 10 player feature vectors + dynamic token stream + next-event labels at each anchor.

**Files:**
- Create: `code/model/dataset.py`
- Create: `data/splits/plan_a_holdout.txt` (generated)
- Test: `code/tests/model/test_dataset.py`

- [ ] **Step 1: Generate the held-out split**

Run this exact Python one-liner to deterministically hash match_ids into train/holdout (90/10):

```bash
cd /home/lunaris/build/howtowin.lol && mkdir -p data/splits && python - <<'PY'
import hashlib, sqlite3
conn = sqlite3.connect("data/howtowin.db")
rows = conn.execute("SELECT match_id FROM games ORDER BY match_id").fetchall()
holdout = []
for (mid,) in rows:
    h = int(hashlib.sha1(mid.encode()).hexdigest(), 16)
    if h % 10 == 0:
        holdout.append(mid)
with open("data/splits/plan_a_holdout.txt", "w") as f:
    for mid in holdout:
        f.write(mid + "\n")
print(f"Holdout size: {len(holdout)} / {len(rows)}")
PY
```

Expected: prints `Holdout size: N / M` where N is ~10% of M.

- [ ] **Step 2: Write the failing test**

```python
# code/tests/model/test_dataset.py
import torch
from model.dataset import MatchDataset, collate_games, load_split


def test_split_loads():
    holdout = load_split("holdout")
    assert isinstance(holdout, list)
    train = load_split("train")
    assert len(set(holdout) & set(train)) == 0


def test_dataset_yields_sample(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    assert len(ds) == 1
    sample = ds[0]
    assert "static" in sample
    assert "players" in sample
    assert "tokens" in sample
    assert "token_actors" in sample
    assert "token_timestamps" in sample
    assert "labels" in sample
    assert "label_mask" in sample
    # Players should be (10, PLAYER_FEATURE_DIM).
    assert sample["players"].shape[0] == 10
    # Tokens are 1-D tensors.
    assert sample["tokens"].dim() == 1
    assert sample["tokens"].shape == sample["token_actors"].shape
    assert sample["tokens"].shape == sample["token_timestamps"].shape


def test_labels_align_with_anchors(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    # Only anchor positions carry supervised labels; others should be masked.
    from model.tokens import ANCHOR_TOKEN
    is_anchor = (sample["tokens"] == ANCHOR_TOKEN)
    # Mask should be 1 exactly at anchor positions (except the final anchor
    # which has no "next" event).
    mask = sample["label_mask"].bool()
    assert mask.sum() <= is_anchor.sum()


def test_collate_batches_games(fixture_match_id):
    ds = MatchDataset([fixture_match_id, fixture_match_id])
    batch = collate_games([ds[0], ds[1]])
    assert batch["static"].shape[0] == 2
    assert batch["players"].shape[0] == 2
    assert batch["tokens"].shape[0] == 2
    # Sequences padded to equal length.
    assert batch["tokens"].shape[1] == batch["labels"].shape[1]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.dataset'`.

- [ ] **Step 4: Implement the dataset**

```python
# code/model/dataset.py
"""PyTorch Dataset wrapping tokenizer + features + labels.

Each sample is one game. Labels are per-anchor: the distribution of
meaningful event types fired in the minute following that anchor
(multi-hot). Training target at each anchor is this multi-hot vector;
top-5 eval asks whether the top-5 predicted classes cover the truth.

Plan A uses multi-hot targets to match "top-5 of next-minute events".
"""
import os
import torch
from torch.utils.data import Dataset

from db import get_conn
from raw_db import get_raw_match
from model.tokenizer import tokenize_match
from model.tokens import (
    ANCHOR_TOKEN, PAD_TOKEN, EVENT_TYPE_TO_ID, NUM_EVENT_TYPES,
)
from model.patch_params import patch_vector_for_match, PATCH_VECTOR_DIM
from model.player_features import player_feature_vector, PLAYER_FEATURE_DIM


SPLIT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'splits')


def load_split(name):
    """name in {'holdout', 'train'}. 'train' = all games minus holdout."""
    holdout_path = os.path.join(SPLIT_DIR, 'plan_a_holdout.txt')
    with open(holdout_path) as f:
        holdout = [line.strip() for line in f if line.strip()]
    if name == 'holdout':
        return holdout
    conn = get_conn()
    try:
        all_ids = [r["match_id"] for r in
                   conn.execute("SELECT match_id FROM games").fetchall()]
    finally:
        conn.close()
    hset = set(holdout)
    return [m for m in all_ids if m not in hset]


def _build_labels(tokens):
    """For each anchor, a multi-hot over NUM_EVENT_TYPES of events in the
    NEXT minute. Final anchor gets mask=0."""
    n = len(tokens)
    labels = torch.zeros(n, NUM_EVENT_TYPES, dtype=torch.float32)
    mask = torch.zeros(n, dtype=torch.float32)
    anchor_indices = [i for i, t in enumerate(tokens) if t.type_id == ANCHOR_TOKEN]
    for k, idx in enumerate(anchor_indices[:-1]):
        next_anchor_idx = anchor_indices[k + 1]
        for j in range(idx + 1, next_anchor_idx):
            t = tokens[j]
            if t.type_id >= 8:  # event tokens start at _RESERVED_COUNT = 8
                evt_offset = t.type_id - 8
                if 0 <= evt_offset < NUM_EVENT_TYPES:
                    labels[idx, evt_offset] = 1.0
        mask[idx] = 1.0
    return labels, mask


class MatchDataset(Dataset):
    def __init__(self, match_ids):
        self.match_ids = list(match_ids)

    def __len__(self):
        return len(self.match_ids)

    def __getitem__(self, i):
        mid = self.match_ids[i]
        tokens = tokenize_match(mid)
        labels, label_mask = _build_labels(tokens)

        token_ids = torch.tensor([t.type_id for t in tokens], dtype=torch.long)
        token_actors = torch.tensor([t.actor_slot for t in tokens], dtype=torch.long)
        token_ts = torch.tensor([t.timestamp_ms for t in tokens], dtype=torch.float32)

        static = torch.tensor(patch_vector_for_match(mid), dtype=torch.float32)

        match, _ = get_raw_match(mid)
        players = torch.zeros(10, PLAYER_FEATURE_DIM, dtype=torch.float32)
        if match:
            for i_p, p in enumerate(match["info"]["participants"][:10]):
                players[i_p] = torch.tensor(player_feature_vector(p["puuid"]), dtype=torch.float32)

        return {
            "static": static,
            "players": players,
            "tokens": token_ids,
            "token_actors": token_actors,
            "token_timestamps": token_ts,
            "labels": labels,
            "label_mask": label_mask,
        }


def collate_games(samples):
    """Pad sequences to the longest in the batch."""
    max_len = max(s["tokens"].shape[0] for s in samples)
    B = len(samples)
    tokens = torch.full((B, max_len), PAD_TOKEN, dtype=torch.long)
    actors = torch.zeros(B, max_len, dtype=torch.long)
    ts = torch.zeros(B, max_len, dtype=torch.float32)
    labels = torch.zeros(B, max_len, NUM_EVENT_TYPES, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.float32)
    key_pad = torch.ones(B, max_len, dtype=torch.bool)  # True = pad

    for b, s in enumerate(samples):
        L = s["tokens"].shape[0]
        tokens[b, :L] = s["tokens"]
        actors[b, :L] = s["token_actors"]
        ts[b, :L] = s["token_timestamps"]
        labels[b, :L] = s["labels"]
        mask[b, :L] = s["label_mask"]
        key_pad[b, :L] = False

    return {
        "static": torch.stack([s["static"] for s in samples]),
        "players": torch.stack([s["players"] for s in samples]),
        "tokens": tokens,
        "token_actors": actors,
        "token_timestamps": ts,
        "labels": labels,
        "label_mask": mask,
        "key_pad_mask": key_pad,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_dataset.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add code/model/dataset.py code/tests/model/test_dataset.py data/splits/plan_a_holdout.txt
git commit -m "plan-a: dataset + holdout split + multi-hot next-minute-event labels"
```

---

## Task 9: Encoder modules

Three encoder `nn.Module` classes that map raw inputs into the transformer's embedding space.

**Files:**
- Create: `code/model/encoders.py`
- Test: `code/tests/model/test_encoders.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_encoders.py
import torch
from model.encoders import (
    StaticContextEncoder, PlayerModelEncoder, DynamicStreamEmbedder,
    D_MODEL,
)
from model.patch_params import PATCH_VECTOR_DIM
from model.player_features import PLAYER_FEATURE_DIM
from model.tokens import VOCAB_SIZE, NUM_SLOTS


def test_static_encoder_output_shape():
    enc = StaticContextEncoder()
    x = torch.randn(4, PATCH_VECTOR_DIM)
    out = enc(x)
    assert out.shape == (4, D_MODEL)


def test_player_encoder_output_shape():
    enc = PlayerModelEncoder(max_puuids=100)
    crafted = torch.randn(4, 10, PLAYER_FEATURE_DIM)
    puuid_ids = torch.zeros(4, 10, dtype=torch.long)  # all "unknown"
    out = enc(crafted, puuid_ids)
    assert out.shape == (4, 10, D_MODEL)


def test_dynamic_embedder_output_shape():
    emb = DynamicStreamEmbedder()
    tokens = torch.zeros(4, 50, dtype=torch.long)
    actors = torch.zeros(4, 50, dtype=torch.long)
    ts = torch.zeros(4, 50)
    player_emb = torch.randn(4, 10, D_MODEL)
    out = emb(tokens, actors, ts, player_emb)
    assert out.shape == (4, 50, D_MODEL)


def test_player_residual_updates_from_known_puuid():
    enc = PlayerModelEncoder(max_puuids=10)
    crafted = torch.randn(1, 1, PLAYER_FEATURE_DIM)
    id_zero = torch.zeros(1, 1, dtype=torch.long)  # unknown (index 0)
    id_one = torch.ones(1, 1, dtype=torch.long)   # known slot 1
    out_zero = enc(crafted, id_zero)
    out_one = enc(crafted, id_one)
    # Different residual -> different output.
    assert not torch.allclose(out_zero, out_one, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_encoders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.encoders'`.

- [ ] **Step 3: Implement encoders**

```python
# code/model/encoders.py
"""Three-stream encoders producing D_MODEL embeddings.

StaticContextEncoder: MLP over dense patch-param + game-context vector.
PlayerModelEncoder:   crafted-features MLP + per-puuid residual embedding.
DynamicStreamEmbedder: token-type + actor-slot + player-fusion + timestamp
                      positional encoding.

The residual embedding table has index 0 reserved for "unknown puuid"
(zero-init, frozen to zero during training).
"""
import math
import torch
import torch.nn as nn

from model.tokens import VOCAB_SIZE, NUM_SLOTS
from model.patch_params import PATCH_VECTOR_DIM
from model.player_features import PLAYER_FEATURE_DIM

D_MODEL = 256


class StaticContextEncoder(nn.Module):
    def __init__(self, input_dim=PATCH_VECTOR_DIM, d_model=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class PlayerModelEncoder(nn.Module):
    def __init__(self, max_puuids, feature_dim=PLAYER_FEATURE_DIM, d_model=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Index 0 is "unknown" (frozen-zero); 1..max_puuids-1 are learnable.
        self.residual = nn.Embedding(max_puuids, d_model, padding_idx=0)
        nn.init.zeros_(self.residual.weight[0])

    def forward(self, crafted, puuid_ids):
        """crafted: (B, 10, feature_dim). puuid_ids: (B, 10)."""
        base = self.mlp(crafted)
        delta = self.residual(puuid_ids)
        return base + delta


class DynamicStreamEmbedder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_slots=NUM_SLOTS, d_model=D_MODEL):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.actor_emb = nn.Embedding(num_slots, d_model)
        self.d_model = d_model

    def _positional(self, ts):
        """Sinusoidal positional encoding from real timestamps (ms)."""
        B, L = ts.shape
        device = ts.device
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        # Normalize timestamps to minutes so sin args are reasonable.
        t = (ts / 60000.0).unsqueeze(-1)
        args = t * div
        pe = torch.zeros(B, L, self.d_model, device=device)
        pe[..., 0::2] = torch.sin(args)
        pe[..., 1::2] = torch.cos(args)
        return pe

    def forward(self, tokens, actors, timestamps, player_emb):
        """tokens, actors: (B, L). timestamps: (B, L). player_emb: (B, 10, D)."""
        tok = self.token_emb(tokens)
        act = self.actor_emb(actors)
        pos = self._positional(timestamps)

        # Fuse player embedding at event positions whose actor is 1..10.
        B, L, D = tok.shape
        # Gather per-token player embedding (zero for actor_slot == 0).
        actor_mask = (actors > 0)  # (B, L)
        actor_safe = actors.clamp(min=1) - 1  # (B, L) in [0, 9]
        player_gather = torch.gather(
            player_emb, 1,
            actor_safe.unsqueeze(-1).expand(B, L, D)
        )
        player_gather = player_gather * actor_mask.unsqueeze(-1).to(player_gather.dtype)

        return tok + act + pos + player_gather
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_encoders.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/model/encoders.py code/tests/model/test_encoders.py
git commit -m "plan-a: three-stream encoder modules (static/player/dynamic)"
```

---

## Task 10: Baseline transformer model

Causal transformer over the dynamic stream, with static context prepended as an extra prefix token and the next-event head reading from anchor positions only.

**Files:**
- Create: `code/model/baseline.py`
- Test: `code/tests/model/test_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_baseline.py
import torch
from model.baseline import CausalTransformerBaseline
from model.encoders import D_MODEL
from model.tokens import NUM_EVENT_TYPES
from model.patch_params import PATCH_VECTOR_DIM
from model.player_features import PLAYER_FEATURE_DIM


def test_forward_shape():
    model = CausalTransformerBaseline(max_puuids=100)
    B, L = 2, 30
    batch = {
        "static": torch.randn(B, PATCH_VECTOR_DIM),
        "players": torch.randn(B, 10, PLAYER_FEATURE_DIM),
        "player_ids": torch.zeros(B, 10, dtype=torch.long),
        "tokens": torch.zeros(B, L, dtype=torch.long),
        "token_actors": torch.zeros(B, L, dtype=torch.long),
        "token_timestamps": torch.zeros(B, L),
        "key_pad_mask": torch.zeros(B, L, dtype=torch.bool),
    }
    logits = model(batch)
    assert logits.shape == (B, L, NUM_EVENT_TYPES)


def test_forward_is_differentiable():
    model = CausalTransformerBaseline(max_puuids=100)
    B, L = 2, 30
    batch = {
        "static": torch.randn(B, PATCH_VECTOR_DIM),
        "players": torch.randn(B, 10, PLAYER_FEATURE_DIM),
        "player_ids": torch.zeros(B, 10, dtype=torch.long),
        "tokens": torch.zeros(B, L, dtype=torch.long),
        "token_actors": torch.zeros(B, L, dtype=torch.long),
        "token_timestamps": torch.zeros(B, L),
        "key_pad_mask": torch.zeros(B, L, dtype=torch.bool),
    }
    logits = model(batch)
    loss = logits.sum()
    loss.backward()
    # At least one parameter has nonzero grad.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.baseline'`.

- [ ] **Step 3: Implement baseline**

```python
# code/model/baseline.py
"""CausalTransformerBaseline: predicts multi-hot next-minute event set at
each anchor position.

Architecture:
  - StaticContextEncoder produces a (B, D) vector prepended as extra token.
  - PlayerModelEncoder produces per-participant (B, 10, D) used by
    DynamicStreamEmbedder to fuse player identity into actor tokens.
  - Causal Transformer over [static_ctx, dynamic_stream].
  - Next-event head reads at each dynamic position -> (B, L, NUM_EVENT_TYPES).
"""
import torch
import torch.nn as nn

from model.encoders import (
    StaticContextEncoder, PlayerModelEncoder, DynamicStreamEmbedder, D_MODEL,
)
from model.tokens import NUM_EVENT_TYPES


class CausalTransformerBaseline(nn.Module):
    def __init__(self, max_puuids, n_layers=6, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.static_enc = StaticContextEncoder()
        self.player_enc = PlayerModelEncoder(max_puuids=max_puuids)
        self.dyn_emb = DynamicStreamEmbedder()

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.head = nn.Linear(D_MODEL, NUM_EVENT_TYPES)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, batch):
        static = self.static_enc(batch["static"])              # (B, D)
        players = self.player_enc(batch["players"], batch["player_ids"])  # (B, 10, D)
        dyn = self.dyn_emb(
            batch["tokens"], batch["token_actors"],
            batch["token_timestamps"], players,
        )  # (B, L, D)

        B, L, D = dyn.shape
        static_tok = static.unsqueeze(1)  # (B, 1, D)
        seq = torch.cat([static_tok, dyn], dim=1)  # (B, 1+L, D)

        # Causal mask over seq; static token at position 0 is always visible.
        mask = self._causal_mask(1 + L, seq.device)

        # Key-padding mask: static never padded; dynamic uses key_pad_mask.
        key_pad_mask = batch.get("key_pad_mask")
        if key_pad_mask is not None:
            pad = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=seq.device),
                key_pad_mask,
            ], dim=1)
        else:
            pad = None

        out = self.transformer(seq, mask=mask, src_key_padding_mask=pad)
        # Drop the static prefix position for the head.
        out = out[:, 1:, :]  # (B, L, D)
        return self.head(out)  # (B, L, NUM_EVENT_TYPES)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_baseline.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/model/baseline.py code/tests/model/test_baseline.py
git commit -m "plan-a: causal transformer baseline with next-event head"
```

---

## Task 11: Training loop

Training loop with multi-label binary cross-entropy loss, checkpointing, and per-epoch validation.

**Files:**
- Create: `code/model/train.py`
- Modify: `code/model/dataset.py` — add `puuid_index` for player residual IDs.
- Test: `code/tests/model/test_integration.py` (a thin overfit smoke test; full M1 eval is Task 14).

- [ ] **Step 1: Add puuid indexing to dataset**

Append to `code/model/dataset.py`:

```python
def build_puuid_index(match_ids, max_puuids=20000):
    """Assign integer IDs 1..max_puuids-1 to the most-frequent puuids in the
    given match_ids. Returns dict puuid -> id. Unknown puuids resolve to 0.
    """
    from collections import Counter
    counts = Counter()
    for mid in match_ids:
        match, _ = get_raw_match(mid)
        if not match:
            continue
        for p in match["info"]["participants"][:10]:
            counts[p["puuid"]] += 1
    ranked = [puuid for puuid, _ in counts.most_common(max_puuids - 1)]
    return {puuid: i + 1 for i, puuid in enumerate(ranked)}


def puuid_ids_for_match(match_id, puuid_index):
    match, _ = get_raw_match(match_id)
    ids = torch.zeros(10, dtype=torch.long)
    if match:
        for i, p in enumerate(match["info"]["participants"][:10]):
            ids[i] = puuid_index.get(p["puuid"], 0)
    return ids
```

Modify `MatchDataset.__init__` to accept `puuid_index` and `__getitem__` to include `player_ids`:

```python
class MatchDataset(Dataset):
    def __init__(self, match_ids, puuid_index=None):
        self.match_ids = list(match_ids)
        self.puuid_index = puuid_index or {}

    def __len__(self):
        return len(self.match_ids)

    def __getitem__(self, i):
        mid = self.match_ids[i]
        tokens = tokenize_match(mid)
        labels, label_mask = _build_labels(tokens)

        token_ids = torch.tensor([t.type_id for t in tokens], dtype=torch.long)
        token_actors = torch.tensor([t.actor_slot for t in tokens], dtype=torch.long)
        token_ts = torch.tensor([t.timestamp_ms for t in tokens], dtype=torch.float32)

        static = torch.tensor(patch_vector_for_match(mid), dtype=torch.float32)

        match, _ = get_raw_match(mid)
        players = torch.zeros(10, PLAYER_FEATURE_DIM, dtype=torch.float32)
        player_ids = torch.zeros(10, dtype=torch.long)
        if match:
            for i_p, p in enumerate(match["info"]["participants"][:10]):
                players[i_p] = torch.tensor(player_feature_vector(p["puuid"]), dtype=torch.float32)
                player_ids[i_p] = self.puuid_index.get(p["puuid"], 0)

        return {
            "static": static,
            "players": players,
            "player_ids": player_ids,
            "tokens": token_ids,
            "token_actors": token_actors,
            "token_timestamps": token_ts,
            "labels": labels,
            "label_mask": label_mask,
        }
```

Update `collate_games` to stack `player_ids`:

```python
def collate_games(samples):
    max_len = max(s["tokens"].shape[0] for s in samples)
    B = len(samples)
    tokens = torch.full((B, max_len), PAD_TOKEN, dtype=torch.long)
    actors = torch.zeros(B, max_len, dtype=torch.long)
    ts = torch.zeros(B, max_len, dtype=torch.float32)
    labels = torch.zeros(B, max_len, NUM_EVENT_TYPES, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.float32)
    key_pad = torch.ones(B, max_len, dtype=torch.bool)

    for b, s in enumerate(samples):
        L = s["tokens"].shape[0]
        tokens[b, :L] = s["tokens"]
        actors[b, :L] = s["token_actors"]
        ts[b, :L] = s["token_timestamps"]
        labels[b, :L] = s["labels"]
        mask[b, :L] = s["label_mask"]
        key_pad[b, :L] = False

    return {
        "static": torch.stack([s["static"] for s in samples]),
        "players": torch.stack([s["players"] for s in samples]),
        "player_ids": torch.stack([s["player_ids"] for s in samples]),
        "tokens": tokens,
        "token_actors": actors,
        "token_timestamps": ts,
        "labels": labels,
        "label_mask": mask,
        "key_pad_mask": key_pad,
    }
```

- [ ] **Step 2: Implement the training loop**

```python
# code/model/train.py
"""Training loop for the baseline classifier."""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dataset import (
    MatchDataset, collate_games, load_split, build_puuid_index,
)
from model.baseline import CausalTransformerBaseline
from model.tokens import NUM_EVENT_TYPES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'model_checkpoints')


def masked_bce_loss(logits, labels, mask):
    """Binary cross-entropy averaged over masked positions only."""
    per_pos = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction='none'
    ).mean(dim=-1)  # (B, L)
    denom = mask.sum().clamp(min=1.0)
    return (per_pos * mask).sum() / denom


def top5_accuracy(logits, labels, mask):
    """Fraction of masked positions where top-5 predicted classes cover the
    majority of the ground-truth positive set. Specifically: position i
    counts as a hit if top-5 predictions include the single most-common
    ground-truth class at that position. If no positives at that position,
    skip."""
    B, L, C = logits.shape
    topk = logits.topk(k=min(5, C), dim=-1).indices  # (B, L, 5)
    top_true = labels.argmax(dim=-1)  # (B, L) — dominant class per position
    pos_mask = (labels.sum(dim=-1) > 0) & mask.bool()
    hits = (topk == top_true.unsqueeze(-1)).any(dim=-1) & pos_mask
    denom = pos_mask.sum().clamp(min=1)
    return hits.sum().float() / denom.float()


def train_loop(
    train_match_ids,
    val_match_ids,
    epochs=20,
    batch_size=8,
    lr=3e-4,
    device=None,
    max_puuids=20000,
    log_every=10,
    checkpoint_tag="baseline",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train: {len(train_match_ids)}  Val: {len(val_match_ids)}")

    puuid_index = build_puuid_index(train_match_ids, max_puuids=max_puuids)
    print(f"Built puuid index with {len(puuid_index)} known puuids")

    train_ds = MatchDataset(train_match_ids, puuid_index=puuid_index)
    val_ds = MatchDataset(val_match_ids, puuid_index=puuid_index)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_games, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_games, num_workers=0,
    )

    model = CausalTransformerBaseline(max_puuids=max_puuids).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_top5 = 0.0

    for epoch in range(epochs):
        model.train()
        ep_start = time.time()
        tot_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = masked_bce_loss(logits, batch["labels"], batch["label_mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            n_batches += 1
            if step % log_every == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_loss = tot_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_top5 = 0.0
            v_batches = 0
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch)
                val_loss += masked_bce_loss(logits, batch["labels"], batch["label_mask"]).item()
                val_top5 += top5_accuracy(logits, batch["labels"], batch["label_mask"]).item()
                v_batches += 1
            val_loss /= max(v_batches, 1)
            val_top5 /= max(v_batches, 1)

        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
              f"val_top5={val_top5:.4f}  ({time.time()-ep_start:.1f}s)")

        if val_top5 > best_val_top5:
            best_val_top5 = val_top5
            ckpt = os.path.join(CHECKPOINT_DIR, f"{checkpoint_tag}_best.pt")
            torch.save({
                "model": model.state_dict(),
                "puuid_index": puuid_index,
                "val_top5": val_top5,
                "epoch": epoch,
            }, ckpt)
            print(f"  saved best checkpoint -> {ckpt}")

    return best_val_top5
```

- [ ] **Step 3: Write integration smoke test**

```python
# code/tests/model/test_integration.py
from model.train import train_loop
from model.dataset import load_split


def test_shakedown_overfit_tiny_corpus():
    """Sanity: the pipeline should overfit a tiny corpus. 5 games, 15 epochs.
    Expect val_top5 > 0.3 which is a low bar but proves signal flows
    end-to-end. The proper M1 check is Task 14."""
    train_ids = load_split("train")[:5]
    val_ids = train_ids  # same games — testing overfit
    best = train_loop(
        train_match_ids=train_ids,
        val_match_ids=val_ids,
        epochs=5,
        batch_size=2,
        lr=3e-4,
        log_every=100,
        checkpoint_tag="shakedown_smoke",
    )
    assert best > 0.1  # very low bar — just proves the loop runs and learns *something*.
```

- [ ] **Step 4: Run the smoke test**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_integration.py -v -s`
Expected: passes within a few minutes on CPU. Expected `val_top5 > 0.1` after 5 epochs on 5 games.

- [ ] **Step 5: Commit**

```bash
git add code/model/train.py code/model/dataset.py code/tests/model/test_integration.py
git commit -m "plan-a: training loop + multi-label BCE + top-5 metric + overfit smoke"
```

---

## Task 12: Eval diagnostics

Per-anchor-minute breakdown of top-5 accuracy, to diagnose where the model is weak.

**Files:**
- Create: `code/model/eval.py`
- Test: `code/tests/model/test_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/model/test_eval.py
import torch
from model.eval import top5_by_minute


def test_top5_by_minute_shape():
    B, L, C = 2, 20, 11
    logits = torch.randn(B, L, C)
    labels = torch.zeros(B, L, C)
    labels[:, :, 0] = 1.0  # all label = class 0
    mask = torch.ones(B, L)
    timestamps = torch.arange(0, L * 60000, 60000, dtype=torch.float32).unsqueeze(0).expand(B, L)
    result = top5_by_minute(logits, labels, mask, timestamps)
    assert "overall" in result
    assert "by_minute" in result
    assert isinstance(result["by_minute"], dict)


def test_top5_perfect_when_class_0_is_argmax():
    B, L, C = 1, 5, 11
    logits = torch.zeros(B, L, C)
    logits[..., 0] = 10.0  # class 0 easily in top-5
    labels = torch.zeros(B, L, C)
    labels[..., 0] = 1.0
    mask = torch.ones(B, L)
    timestamps = torch.zeros(B, L)
    result = top5_by_minute(logits, labels, mask, timestamps)
    assert result["overall"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.eval'`.

- [ ] **Step 3: Implement eval**

```python
# code/model/eval.py
"""Evaluation helpers beyond per-batch top-5."""
import torch


def top5_by_minute(logits, labels, mask, timestamps, k=5):
    """Return dict with 'overall' top-k accuracy and 'by_minute' breakdown.

    logits: (B, L, C)
    labels: (B, L, C) multi-hot
    mask:   (B, L)
    timestamps: (B, L) in ms
    """
    B, L, C = logits.shape
    topk = logits.topk(k=min(k, C), dim=-1).indices
    top_true = labels.argmax(dim=-1)
    pos_mask = (labels.sum(dim=-1) > 0) & mask.bool()
    hits = (topk == top_true.unsqueeze(-1)).any(dim=-1) & pos_mask

    minutes = (timestamps / 60000.0).long()

    result = {"overall": 0.0, "by_minute": {}}
    flat_hits = hits[pos_mask]
    if flat_hits.numel() > 0:
        result["overall"] = flat_hits.float().mean().item()

    unique_minutes = torch.unique(minutes[pos_mask])
    for m in unique_minutes.tolist():
        m_mask = (minutes == m) & pos_mask
        if m_mask.any():
            acc = hits[m_mask].float().mean().item()
            result["by_minute"][m] = acc

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m pytest tests/model/test_eval.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/model/eval.py code/tests/model/test_eval.py
git commit -m "plan-a: top-5 eval with by-minute diagnostic breakdown"
```

---

## Task 13: CLI entry points

Unified `python -m model.cli <shakedown|train|eval>` entry points.

**Files:**
- Create: `code/model/cli.py`

- [ ] **Step 1: Implement CLI**

```python
# code/model/cli.py
"""Command-line entry points for Plan A.

Usage:
  python -m model.cli shakedown          # M1 overfit-tiny-corpus check
  python -m model.cli train [--epochs N] # full train on train split
  python -m model.cli eval               # M2 held-out evaluation using best ckpt
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import (
    MatchDataset, collate_games, load_split, build_puuid_index,
)
from model.train import train_loop, CHECKPOINT_DIR, masked_bce_loss
from model.baseline import CausalTransformerBaseline
from model.eval import top5_by_minute
from torch.utils.data import DataLoader


def cmd_shakedown():
    """M1 acceptance: overfit a 50-game corpus near-zero loss."""
    train_ids = load_split("train")[:50]
    best = train_loop(
        train_match_ids=train_ids,
        val_match_ids=train_ids,
        epochs=30,
        batch_size=4,
        lr=3e-4,
        log_every=20,
        checkpoint_tag="shakedown",
    )
    print(f"\nSHAKEDOWN best train-set top5: {best:.4f}")
    print("M1 acceptance: best >= 0.9 (near-overfit on 50 games)")
    return best >= 0.9


def cmd_train(args):
    train_ids = load_split("train")
    val_ids = load_split("holdout")
    best = train_loop(
        train_match_ids=train_ids,
        val_match_ids=val_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=50,
        checkpoint_tag="baseline_full",
    )
    print(f"\nFULL TRAIN best val top5: {best:.4f}")


def cmd_eval():
    """M2 acceptance: top-5 >= 0.95 (stretch target) on held-out."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "baseline_full_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    puuid_index = ckpt["puuid_index"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CausalTransformerBaseline(max_puuids=len(puuid_index) + 1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ids = load_split("holdout")
    val_ds = MatchDataset(val_ids, puuid_index=puuid_index)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_games)

    all_results = {"overall": 0.0, "by_minute": {}}
    n_batches = 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            r = top5_by_minute(
                logits, batch["labels"], batch["label_mask"], batch["token_timestamps"],
            )
            all_results["overall"] += r["overall"]
            for m, acc in r["by_minute"].items():
                all_results["by_minute"].setdefault(m, []).append(acc)
            n_batches += 1

    overall = all_results["overall"] / max(n_batches, 1)
    print(f"\nHELD-OUT top-5 (overall): {overall:.4f}")
    print("By minute:")
    for m in sorted(all_results["by_minute"].keys()):
        accs = all_results["by_minute"][m]
        print(f"  min {m:3d}: {sum(accs)/len(accs):.4f} (n={len(accs)})")
    print(f"\nM2 stretch target: overall >= 0.95")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shakedown")
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--epochs", type=int, default=30)
    p_tr.add_argument("--batch_size", type=int, default=8)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    sub.add_parser("eval")
    args = parser.parse_args()

    if args.cmd == "shakedown":
        ok = cmd_shakedown()
        sys.exit(0 if ok else 1)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval()
```

- [ ] **Step 2: Verify imports**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -c "from model import cli; print('cli module imports OK')"`
Expected: prints the OK message.

- [ ] **Step 3: Commit**

```bash
git add code/model/cli.py
git commit -m "plan-a: CLI entry points (shakedown, train, eval)"
```

---

## Task 14: M1 shakedown — acceptance

Run the pipeline shakedown. Proves end-to-end wiring by overfitting a 50-game subset.

**Files:**
- None new — this task runs existing code and records the result.

- [ ] **Step 1: Run shakedown**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m model.cli shakedown 2>&1 | tee /tmp/shakedown.log`
Expected: completes in 5–30 minutes on CPU. Final line should read `M1 acceptance: best >= 0.9 (near-overfit on 50 games)` with `best` ≥ 0.9.

- [ ] **Step 2: Diagnose if it fails**

If `best` < 0.9:
- Check training-loss curve in the log. If loss is stuck: increase `epochs` in `cmd_shakedown` to 60 and retry.
- If loss drops but top-5 stays low: the label construction may be wrong. Inspect `_build_labels` in `dataset.py` — confirm anchor indices and next-minute aggregation.
- If loss is NaN: lower `lr` from 3e-4 to 1e-4 in `cmd_shakedown`.

Do not advance past this task until shakedown passes.

- [ ] **Step 3: Record milestone**

Append a short line to the plan status in `docs/superpowers/specs/2026-04-15-sequence-model-design.md` under a new `## Status` section:

```markdown
## Status

- 2026-04-15: Plan A Milestone 1 (pipeline shakedown) passed. Overfit top-5 = <value>.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-15-sequence-model-design.md
git commit -m "plan-a: M1 shakedown passed (overfit top-5 recorded)"
```

---

## Task 15: M2 full train + held-out eval — acceptance

Train on the full train split and measure held-out top-5.

**Files:**
- None new.

- [ ] **Step 1: Launch full training run**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m model.cli train --epochs 40 2>&1 | tee /tmp/train.log`
Expected: runs to completion. Duration depends on corpus size and hardware — a few hours on a single GPU, overnight on CPU for ~700 games.

- [ ] **Step 2: Held-out evaluation**

Run: `cd /home/lunaris/build/howtowin.lol/code && uv run python -m model.cli eval 2>&1 | tee /tmp/eval.log`
Expected: prints overall top-5 and per-minute breakdown.

- [ ] **Step 3: Interpret the result**

- **If overall ≥ 0.95**: stretch target hit. Record and move on.
- **If 0.80 ≤ overall < 0.95**: strong baseline, below stretch target. Record the gap; this is still acceptance for Plan A since the spec explicitly marks 0.95 as a stretch target. Plan B (RSSM) then has a real bar to beat.
- **If overall < 0.80**: investigate before accepting. Likely causes: corpus too small (scale up seeding before retraining), label noise (revisit `_build_labels`), or underfit (more epochs, larger model).

- [ ] **Step 4: Record result in spec status**

Append to the `## Status` section in `docs/superpowers/specs/2026-04-15-sequence-model-design.md`:

```markdown
- 2026-04-15: Plan A Milestone 2 (baseline classifier) complete. Held-out top-5 = <value>. Per-minute: <paste the ordered list>.
```

- [ ] **Step 5: Commit the results**

```bash
git add docs/superpowers/specs/2026-04-15-sequence-model-design.md
git commit -m "plan-a: M2 baseline held-out top-5 recorded"
```

- [ ] **Step 6: Plan A retrospective**

Write a brief retrospective into `docs/superpowers/plans/2026-04-15-plan-a-retro.md`:

```markdown
# Plan A Retrospective

## Final numbers
- M1 overfit top-5 (50 games): <value>
- M2 held-out top-5: <value>
- Per-minute breakdown: ...

## What worked
- ...

## What surprised us
- ...

## Implications for Plan B
- RSSM must at least match M2 top-5 on next-event (or we have a bug).
- Target gaps to address: ...

## Data pipeline notes for future plans
- puuid-index size: ...
- train/holdout split: ...
- dragon cache status: ...
```

Fill in the bullets with actual observations from training. Commit:

```bash
git add docs/superpowers/plans/2026-04-15-plan-a-retro.md
git commit -m "plan-a: retrospective"
```

---

## Self-Review Notes

1. **Spec coverage**

   | Spec requirement | Task |
   | --- | --- |
   | Static context stream (picks, sides, patch, queue, region, time-of-day, patch-as-numeric) | Task 6 (patch_params), Task 9 (StaticContextEncoder) |
   | Player models stream (crafted + learnable residual) | Task 7 (player_features), Task 9 (PlayerModelEncoder) |
   | Dynamic sequence stream (hybrid time: anchors + events) | Tasks 1, 5, 9 |
   | Decisions (purchase, skill-up, ward — explicit; recall, engage/disengage — inferred) | Tasks 3, 4, 5 |
   | Milestone 1 (pipeline shakedown, overfit tiny corpus) | Task 14 |
   | Milestone 2 (baseline classifier, stretch top-5 ≥ 95%) | Task 15 |

   Not covered here by design (deferred to Plan B+): RSSM, frame/outcome/decision heads, retrieval, lessons. This matches the Plan A scope statement.

2. **Placeholder scan**

   No `TBD`/`TODO`. Every step has concrete code or commands. `{{FIXTURE_MATCH_ID}}` in Task 2 is explicitly a fill-in step with instructions for how to produce the value.

3. **Type consistency**

   - `Token(type_id, actor_slot, target_slot, timestamp_ms)` — used consistently across Tasks 1, 5, 8.
   - `PATCH_VECTOR_DIM`, `PLAYER_FEATURE_DIM`, `D_MODEL` — defined once and imported where needed.
   - `build_puuid_index` / `puuid_ids_for_match` added in Task 11 and consumed by `MatchDataset.__init__` (also updated in Task 11).
   - `batch` dict keys stable: `static`, `players`, `player_ids`, `tokens`, `token_actors`, `token_timestamps`, `labels`, `label_mask`, `key_pad_mask`. The baseline model (Task 10) reads these exact keys; collate (Task 8/11) produces them.

   Minor nit noticed during review: Task 8's initial `collate_games` did not yet include `player_ids`, and Task 10's test synthesizes it directly. Task 11 rewrites `collate_games` to include `player_ids`. Order is correct: Task 10's tests build the batch dict manually, so they don't depend on collate. Task 15 uses the post-Task-11 collate, which does include `player_ids`. No consistency bug.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-15-plan-a-data-pipeline-and-baseline-classifier.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration. Best for a plan this size (15 tasks) since each task is self-contained and the reviewer gap keeps quality high.

2. **Inline Execution** — execute tasks in this session using the executing-plans skill, batch execution with checkpoints for review. Best if you want to watch the work happen and catch design issues as they surface.

Which approach?
