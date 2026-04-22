"""match_id -> ordered token stream with typed payload fields.

Stream layout:
  - One ANCHOR token per frame timestamp.
  - One event token per meaningful event at its timestamp_ms.
  - Inferred RECALL / ENGAGE / DISENGAGE tokens from decisions.py.

The stream is globally sorted by timestamp_ms. Anchors preceding events at
the same timestamp come first.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from db import get_conn
from model.decisions import infer_engages, infer_recalls
from model.tokens import (
    ANCHOR_TOKEN,
    BUILDING_TYPE_TO_ID,
    EVENT_TYPE_TO_ID,
    LANE_TYPE_TO_ID,
    MAX_ITEM_ID,
    MONSTER_SUBTYPE_TO_ID,
    MONSTER_TYPE_TO_ID,
    TOWER_TYPE_TO_ID,
    WARD_TYPE_TO_ID,
    lookup_with_unknown,
)


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
    item_id: int = 0
    skill_slot: int = 0
    monster_type_id: int = 0
    monster_subtype_id: int = 0
    building_type_id: int = 0
    lane_type_id: int = 0
    tower_type_id: int = 0
    ward_type_id: int = 0


def _load_events(conn, match_id):
    rows = conn.execute(
        """SELECT timestamp_ms, event_type, participant_id, killer_id, victim_id,
                  killer_team, team_id, position_x, position_y, details
           FROM events WHERE match_id = ? ORDER BY timestamp_ms""",
        (match_id,)
    ).fetchall()
    out = []
    for row in rows:
        event = dict(row)
        raw_details = event.get("details")
        if raw_details:
            try:
                event["details"] = json.loads(raw_details)
            except json.JSONDecodeError:
                event["details"] = {}
        else:
            event["details"] = {}
        out.append(event)
    return out


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
    if etype in {"CHAMPION_KILL", "BUILDING_KILL", "ELITE_MONSTER_KILL", "CHAMPION_SPECIAL_KILL"}:
        return ev["killer_id"] or 0
    if etype in {"ITEM_PURCHASED", "SKILL_LEVEL_UP", "WARD_PLACED", "WARD_KILL"}:
        return ev["participant_id"] or 0
    return 0


def _target_of(ev):
    if ev["event_type"] == "CHAMPION_KILL":
        return ev["victim_id"] or 0
    return 0


def _payload_kwargs(ev):
    details = ev.get("details") or {}
    etype = ev["event_type"]
    if etype == "ITEM_PURCHASED":
        item_id = int(details.get("itemId", 0) or 0)
        if item_id < 0 or item_id > MAX_ITEM_ID:
            item_id = 0
        return {"item_id": item_id}
    if etype == "SKILL_LEVEL_UP":
        slot = int(details.get("skillSlot", 0) or 0)
        if slot < 0 or slot > 4:
            slot = 0
        return {"skill_slot": slot}
    if etype == "ELITE_MONSTER_KILL":
        return {
            "monster_type_id": lookup_with_unknown(MONSTER_TYPE_TO_ID, details.get("monsterType")),
            "monster_subtype_id": lookup_with_unknown(
                MONSTER_SUBTYPE_TO_ID, details.get("monsterSubType") or details.get("monsterType")
            ),
        }
    if etype == "BUILDING_KILL":
        return {
            "building_type_id": lookup_with_unknown(BUILDING_TYPE_TO_ID, details.get("buildingType")),
            "lane_type_id": lookup_with_unknown(LANE_TYPE_TO_ID, details.get("laneType")),
            "tower_type_id": lookup_with_unknown(TOWER_TYPE_TO_ID, details.get("towerType")),
        }
    if etype in {"WARD_PLACED", "WARD_KILL"}:
        return {"ward_type_id": lookup_with_unknown(WARD_TYPE_TO_ID, details.get("wardType"))}
    return {}


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
            continue  # world-owned events skipped for now
        tokens.append(Token(
            type_id=type_id,
            actor_slot=actor,
            target_slot=target,
            timestamp_ms=ev["timestamp_ms"],
            **_payload_kwargs(ev),
        ))

    # Inferred decisions.
    for r in infer_recalls(frames_by_ts):
        tokens.append(Token(EVENT_TYPE_TO_ID["RECALL"], r["slot"], 0, r["timestamp_ms"]))

    kill_events = [ev for ev in events if ev["event_type"] == "CHAMPION_KILL" and ev["position_x"] is not None]
    for tag in infer_engages(kill_events, frames_by_ts):
        anchor_slot = 1 if tag["team_id"] == 100 else 6
        tokens.append(Token(EVENT_TYPE_TO_ID[tag["event_type"]], anchor_slot, 0, tag["timestamp_ms"]))

    # Sort: anchors before events at the same ts (ANCHOR_TOKEN == 0, min id).
    tokens.sort(key=lambda t: (t.timestamp_ms, t.type_id))
    return tokens
