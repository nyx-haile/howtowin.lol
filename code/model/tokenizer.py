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
