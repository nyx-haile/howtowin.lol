"""PyTorch Dataset wrapping tokenizer + features + labels.

Each sample is one game. For baseline compatibility we still emit the original
multi-hot next-minute event-type labels (`labels`, `label_mask`). Plan B adds a
richer per-anchor observation bundle and factorized next-event labels used by the
world-model path.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from db import get_conn
from raw_db import get_raw_match
from model.cold_holdout import load_player_cold_holdout
from model.patch_params import PATCH_VECTOR_DIM
from model.player_features import PLAYER_FEATURE_DIM, player_feature_vector
from model.static_features import STATIC_VECTOR_DIM, build_static_feature_vector
from model.tokenizer import tokenize_match
from model.tokens import (
    ANCHOR_TOKEN,
    EVENT_TYPE_TO_ID,
    NO_EVENT_LABEL,
    NUM_EVENT_TYPES,
    PAD_TOKEN,
    event_label_from_type_id,
)

# Decision types for next-decision head.
DECISION_MAP = {
    EVENT_TYPE_TO_ID.get("ITEM_PURCHASED", -1): 0,
    EVENT_TYPE_TO_ID.get("SKILL_LEVEL_UP", -1): 1,
    EVENT_TYPE_TO_ID.get("WARD_PLACED", -1): 2,
    EVENT_TYPE_TO_ID.get("RECALL", -1): 3,
    EVENT_TYPE_TO_ID.get("ENGAGE", -1): 4,
    EVENT_TYPE_TO_ID.get("DISENGAGE", -1): 5,
}
DECISION_MAP.pop(-1, None)
NO_DECISION = 6

PARTICIPANT_FRAME_KEYS = (
    "total_gold",
    "current_gold",
    "xp",
    "level",
    "cs",
    "jungle_cs",
    "kills",
    "deaths",
    "assists",
    "pos_x",
    "pos_y",
)
FRAME_FEAT_DIM = len(PARTICIPANT_FRAME_KEYS)
FRAME_FEAT_SCALE = np.array(
    [5000.0, 3000.0, 5000.0, 18.0, 200.0, 120.0, 20.0, 20.0, 30.0, 15000.0, 15000.0],
    dtype=np.float32,
)

MACRO_FEATURE_KEYS = (
    "gold_diff",
    "xp_diff",
    "kill_diff",
    "tower_diff",
    "plate_diff",
    "dragon_diff",
    "baron_diff",
    "herald_diff",
    "grub_diff",
    "blue_dragon_count",
    "red_dragon_count",
    "baron_present",
    "blue_centroid_x",
    "blue_centroid_y",
    "blue_spread",
    "red_centroid_x",
    "red_centroid_y",
    "red_spread",
)
MACRO_FEAT_DIM = len(MACRO_FEATURE_KEYS)
MACRO_FEAT_SCALE = np.array(
    [25000.0, 25000.0, 30.0, 11.0, 15.0, 6.0, 3.0, 3.0, 10.0, 6.0, 6.0, 1.0,
     15000.0, 15000.0, 8000.0, 15000.0, 15000.0, 8000.0],
    dtype=np.float32,
)

SPLIT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'splits')


def load_split(name):
    """name in {'holdout', 'train', 'cold'}."""
    holdout_path = os.path.join(SPLIT_DIR, 'plan_a_holdout.txt')
    with open(holdout_path) as f:
        holdout = [line.strip() for line in f if line.strip()]

    if name == 'cold':
        return sorted(load_player_cold_holdout())

    if name == 'holdout':
        return holdout

    conn = get_conn()
    try:
        all_ids = [r["match_id"] for r in conn.execute("SELECT match_id FROM games").fetchall()]
    finally:
        conn.close()
    hset = set(holdout)
    try:
        player_cold = load_player_cold_holdout()
    except FileNotFoundError:
        player_cold = set()
    return [m for m in all_ids if m not in hset and m not in player_cold]


def _build_labels(tokens):
    """Legacy Plan A labels: per-anchor multi-hot next-minute event-type set."""
    n = len(tokens)
    labels = torch.zeros(n, NUM_EVENT_TYPES, dtype=torch.float32)
    mask = torch.zeros(n, dtype=torch.float32)
    anchor_indices = [i for i, t in enumerate(tokens) if t.type_id == ANCHOR_TOKEN]
    for k, idx in enumerate(anchor_indices[:-1]):
        next_anchor_idx = anchor_indices[k + 1]
        for j in range(idx + 1, next_anchor_idx):
            t = tokens[j]
            if t.type_id >= 8:
                evt_offset = t.type_id - 8
                if 0 <= evt_offset < NUM_EVENT_TYPES:
                    labels[idx, evt_offset] = 1.0
        mask[idx] = 1.0
    return labels, mask


def _team_centroid_and_spread(rows):
    if not rows:
        return 0.0, 0.0, 0.0
    xs = [float(r["pos_x"] or 0.0) for r in rows]
    ys = [float(r["pos_y"] or 0.0) for r in rows]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    spread = (sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys)) / len(xs)) ** 0.5
    return cx, cy, spread


def _update_objective_counters(counters, event):
    details = event.get("details") or {}
    etype = event["event_type"]
    if etype == "ELITE_MONSTER_KILL":
        killer_team = event.get("killer_team") or details.get("killerTeamId")
        if killer_team in (100, 200):
            monster_type = str(details.get("monsterType") or "UNKNOWN").upper()
            if monster_type == "DRAGON":
                counters["dragon"][killer_team] += 1
            elif monster_type == "BARON_NASHOR":
                counters["baron"][killer_team] += 1
            elif monster_type == "RIFTHERALD":
                counters["herald"][killer_team] += 1
            elif monster_type == "HORDE":
                counters["grub"][killer_team] += 1
    elif etype == "BUILDING_KILL":
        if str(details.get("buildingType") or "").upper() == "TOWER_BUILDING":
            lost_team = event.get("team_id") or details.get("teamId")
            if lost_team == 100:
                counters["tower"][200] += 1
            elif lost_team == 200:
                counters["tower"][100] += 1
    elif etype == "TURRET_PLATE_DESTROYED":
        lost_team = event.get("team_id") or details.get("teamId")
        if lost_team == 100:
            counters["plate"][200] += 1
        elif lost_team == 200:
            counters["plate"][100] += 1


class MatchDataset(Dataset):
    def __init__(self, match_ids, puuid_index=None, exclude_match_ids=None, cache_size=8000):
        self.match_ids = list(match_ids)
        self.puuid_index = puuid_index or {}
        self.exclude_match_ids = set(exclude_match_ids) if exclude_match_ids else set()
        self._cache = OrderedDict()
        self._cache_max = min(len(self.match_ids), cache_size)
        self._player_feat = self._preload_player_features()

    def _preload_player_features(self):
        if not self.match_ids:
            return {}
        conn = get_conn()
        try:
            puuids = set()
            chunk_size = 900
            for i in range(0, len(self.match_ids), chunk_size):
                chunk = self.match_ids[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT DISTINCT puuid FROM frames WHERE match_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                puuids.update(r["puuid"] for r in rows)
        finally:
            conn.close()
        return {
            puuid: player_feature_vector(puuid, exclude_match_ids=self.exclude_match_ids)
            for puuid in puuids
        }

    def __len__(self):
        return len(self.match_ids)

    def __getitem__(self, i):
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        sample = self._load(i)
        self._cache[i] = sample
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return sample

    def _load(self, i):
        mid = self.match_ids[i]
        tokens = tokenize_match(mid)
        labels, label_mask = _build_labels(tokens)

        token_ids = torch.tensor([t.type_id for t in tokens], dtype=torch.long)
        token_actors = torch.tensor([t.actor_slot for t in tokens], dtype=torch.long)
        token_targets = torch.tensor([t.target_slot for t in tokens], dtype=torch.long)
        token_ts = torch.tensor([t.timestamp_ms for t in tokens], dtype=torch.float32)
        token_item_ids = torch.tensor([t.item_id for t in tokens], dtype=torch.long)
        token_skill_slots = torch.tensor([t.skill_slot for t in tokens], dtype=torch.long)
        token_monster_types = torch.tensor([t.monster_type_id for t in tokens], dtype=torch.long)
        token_monster_subtypes = torch.tensor([t.monster_subtype_id for t in tokens], dtype=torch.long)
        token_building_types = torch.tensor([t.building_type_id for t in tokens], dtype=torch.long)
        token_lane_types = torch.tensor([t.lane_type_id for t in tokens], dtype=torch.long)
        token_tower_types = torch.tensor([t.tower_type_id for t in tokens], dtype=torch.long)
        token_ward_types = torch.tensor([t.ward_type_id for t in tokens], dtype=torch.long)

        static = torch.tensor(build_static_feature_vector(mid), dtype=torch.float32)

        match, _ = get_raw_match(mid)
        players = torch.zeros(10, PLAYER_FEATURE_DIM, dtype=torch.float32)
        player_ids = torch.zeros(10, dtype=torch.long)
        if match:
            for i_p, p in enumerate(match["info"]["participants"][:10]):
                feat = self._player_feat.get(p["puuid"])
                if feat is not None:
                    players[i_p] = torch.from_numpy(feat)
                player_ids[i_p] = self.puuid_index.get(p["puuid"], 0)

        anchor_idx = [j for j, tok in enumerate(tokens) if tok.type_id == ANCHOR_TOKEN]
        anchor_ts = [tokens[j].timestamp_ms for j in anchor_idx]

        conn = get_conn()
        try:
            all_frame_rows = conn.execute(
                "SELECT timestamp_ms, participant_slot, team_id, current_gold, total_gold, xp, level, cs, jungle_cs, pos_x, pos_y, kills, deaths, assists "
                "FROM frames WHERE match_id = ? AND participant_slot BETWEEN 1 AND 10",
                (mid,),
            ).fetchall()
            all_event_rows = conn.execute(
                "SELECT timestamp_ms, event_type, killer_team, team_id, details "
                "FROM events WHERE match_id = ? ORDER BY timestamp_ms",
                (mid,),
            ).fetchall()
            game_row = conn.execute(
                "SELECT winning_team FROM games WHERE match_id = ?",
                (mid,),
            ).fetchone()
        finally:
            conn.close()

        ts_to_slots = {}
        for r in all_frame_rows:
            ts_to_slots.setdefault(r["timestamp_ms"], {})[r["participant_slot"]] = r

        frame_feats = np.zeros((len(anchor_idx), 10, FRAME_FEAT_DIM), dtype=np.float32)
        macro_feats = np.zeros((len(anchor_idx), MACRO_FEAT_DIM), dtype=np.float32)

        parsed_events = []
        for row in all_event_rows:
            details = row["details"]
            try:
                parsed = json.loads(details) if details else {}
            except json.JSONDecodeError:
                parsed = {}
            parsed_events.append({
                "timestamp_ms": row["timestamp_ms"],
                "event_type": row["event_type"],
                "killer_team": row["killer_team"],
                "team_id": row["team_id"],
                "details": parsed,
            })

        counters = {
            "tower": {100: 0, 200: 0},
            "plate": {100: 0, 200: 0},
            "dragon": {100: 0, 200: 0},
            "baron": {100: 0, 200: 0},
            "herald": {100: 0, 200: 0},
            "grub": {100: 0, 200: 0},
        }
        event_ptr = 0
        for ai, (a_i, ts) in enumerate(zip(anchor_idx, anchor_ts)):
            while event_ptr < len(parsed_events) and parsed_events[event_ptr]["timestamp_ms"] <= ts:
                _update_objective_counters(counters, parsed_events[event_ptr])
                event_ptr += 1

            slot_map = ts_to_slots.get(tokens[a_i].timestamp_ms, {})
            blue_rows, red_rows = [], []
            blue_gold = blue_xp = blue_kills = 0.0
            red_gold = red_xp = red_kills = 0.0
            for slot in range(1, 11):
                r = slot_map.get(slot)
                if r is None:
                    continue
                feat_row = np.array([
                    float(r["total_gold"] or 0),
                    float(r["current_gold"] or 0),
                    float(r["xp"] or 0),
                    float(r["level"] or 0),
                    float(r["cs"] or 0),
                    float(r["jungle_cs"] or 0),
                    float(r["kills"] or 0),
                    float(r["deaths"] or 0),
                    float(r["assists"] or 0),
                    float(r["pos_x"] or 0),
                    float(r["pos_y"] or 0),
                ], dtype=np.float32)
                frame_feats[ai, slot - 1] = feat_row
                if r["team_id"] == 100:
                    blue_rows.append(r)
                    blue_gold += float(r["total_gold"] or 0)
                    blue_xp += float(r["xp"] or 0)
                    blue_kills += float(r["kills"] or 0)
                else:
                    red_rows.append(r)
                    red_gold += float(r["total_gold"] or 0)
                    red_xp += float(r["xp"] or 0)
                    red_kills += float(r["kills"] or 0)

            blue_cx, blue_cy, blue_spread = _team_centroid_and_spread(blue_rows)
            red_cx, red_cy, red_spread = _team_centroid_and_spread(red_rows)
            tower_diff = counters["tower"][100] - counters["tower"][200]
            plate_diff = counters["plate"][100] - counters["plate"][200]
            dragon_diff = counters["dragon"][100] - counters["dragon"][200]
            baron_diff = counters["baron"][100] - counters["baron"][200]
            herald_diff = counters["herald"][100] - counters["herald"][200]
            grub_diff = counters["grub"][100] - counters["grub"][200]
            baron_present = 1.0 if (counters["baron"][100] + counters["baron"][200]) > 0 else 0.0
            macro_feats[ai] = np.array([
                blue_gold - red_gold,
                blue_xp - red_xp,
                blue_kills - red_kills,
                tower_diff,
                plate_diff,
                dragon_diff,
                baron_diff,
                herald_diff,
                grub_diff,
                counters["dragon"][100],
                counters["dragon"][200],
                baron_present,
                blue_cx,
                blue_cy,
                blue_spread,
                red_cx,
                red_cy,
                red_spread,
            ], dtype=np.float32)

        frame_feats /= FRAME_FEAT_SCALE[np.newaxis, np.newaxis, :]
        macro_feats /= MACRO_FEAT_SCALE[np.newaxis, :]
        outcome = 1 if game_row and game_row["winning_team"] == 100 else 0

        decision_labels = np.full((len(anchor_idx), 10), NO_DECISION, dtype=np.int64)
        next_event_type_labels = np.full((len(anchor_idx),), NO_EVENT_LABEL, dtype=np.int64)
        next_event_actor_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_target_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_item_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_skill_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_monster_type_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_monster_subtype_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_building_type_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_lane_type_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_tower_type_labels = np.zeros((len(anchor_idx),), dtype=np.int64)
        next_event_ward_type_labels = np.zeros((len(anchor_idx),), dtype=np.int64)

        window_positions = []
        for ti, a_i in enumerate(anchor_idx):
            nxt = anchor_idx[ti + 1] if ti + 1 < len(anchor_idx) else len(tokens)
            positions = list(range(a_i + 1, nxt))
            window_positions.append(positions)

            first_event = None
            for pos in positions:
                tok = tokens[pos]
                if tok.type_id >= 8:
                    first_event = tok
                    break
            if first_event is not None:
                next_event_type_labels[ti] = event_label_from_type_id(first_event.type_id)
                next_event_actor_labels[ti] = int(first_event.actor_slot)
                next_event_target_labels[ti] = int(first_event.target_slot)
                next_event_item_labels[ti] = int(first_event.item_id)
                next_event_skill_labels[ti] = int(first_event.skill_slot)
                next_event_monster_type_labels[ti] = int(first_event.monster_type_id)
                next_event_monster_subtype_labels[ti] = int(first_event.monster_subtype_id)
                next_event_building_type_labels[ti] = int(first_event.building_type_id)
                next_event_lane_type_labels[ti] = int(first_event.lane_type_id)
                next_event_tower_type_labels[ti] = int(first_event.tower_type_id)
                next_event_ward_type_labels[ti] = int(first_event.ward_type_id)

            for tok in tokens[a_i + 1:nxt]:
                if tok.type_id in DECISION_MAP and 1 <= tok.actor_slot <= 10:
                    decision_labels[ti, tok.actor_slot - 1] = DECISION_MAP[tok.type_id]

        return {
            "static": static,
            "players": players,
            "player_ids": player_ids,
            "tokens": token_ids,
            "token_actors": token_actors,
            "token_targets": token_targets,
            "token_timestamps": token_ts,
            "token_item_ids": token_item_ids,
            "token_skill_slots": token_skill_slots,
            "token_monster_types": token_monster_types,
            "token_monster_subtypes": token_monster_subtypes,
            "token_building_types": token_building_types,
            "token_lane_types": token_lane_types,
            "token_tower_types": token_tower_types,
            "token_ward_types": token_ward_types,
            "labels": labels,
            "label_mask": label_mask,
            "anchor_positions": np.array(anchor_idx, dtype=np.int64),
            "frame_features": frame_feats,
            "anchor_macro_features": macro_feats,
            "decision_labels": decision_labels,
            "next_event_type_labels": next_event_type_labels,
            "next_event_actor_labels": next_event_actor_labels,
            "next_event_target_labels": next_event_target_labels,
            "next_event_item_labels": next_event_item_labels,
            "next_event_skill_labels": next_event_skill_labels,
            "next_event_monster_type_labels": next_event_monster_type_labels,
            "next_event_monster_subtype_labels": next_event_monster_subtype_labels,
            "next_event_building_type_labels": next_event_building_type_labels,
            "next_event_lane_type_labels": next_event_lane_type_labels,
            "next_event_tower_type_labels": next_event_tower_type_labels,
            "next_event_ward_type_labels": next_event_ward_type_labels,
            "window_positions": window_positions,
            "outcome": outcome,
        }


def collate_games(samples):
    max_len = max(s["tokens"].shape[0] for s in samples)
    B = len(samples)
    tokens = torch.full((B, max_len), PAD_TOKEN, dtype=torch.long)
    actors = torch.zeros(B, max_len, dtype=torch.long)
    targets = torch.zeros(B, max_len, dtype=torch.long)
    ts = torch.zeros(B, max_len, dtype=torch.float32)
    item_ids = torch.zeros(B, max_len, dtype=torch.long)
    skill_slots = torch.zeros(B, max_len, dtype=torch.long)
    monster_types = torch.zeros(B, max_len, dtype=torch.long)
    monster_subtypes = torch.zeros(B, max_len, dtype=torch.long)
    building_types = torch.zeros(B, max_len, dtype=torch.long)
    lane_types = torch.zeros(B, max_len, dtype=torch.long)
    tower_types = torch.zeros(B, max_len, dtype=torch.long)
    ward_types = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.zeros(B, max_len, NUM_EVENT_TYPES, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.float32)
    key_pad = torch.ones(B, max_len, dtype=torch.bool)

    for b, s in enumerate(samples):
        L = s["tokens"].shape[0]
        tokens[b, :L] = s["tokens"]
        actors[b, :L] = s["token_actors"]
        targets[b, :L] = s["token_targets"]
        ts[b, :L] = s["token_timestamps"]
        item_ids[b, :L] = s["token_item_ids"]
        skill_slots[b, :L] = s["token_skill_slots"]
        monster_types[b, :L] = s["token_monster_types"]
        monster_subtypes[b, :L] = s["token_monster_subtypes"]
        building_types[b, :L] = s["token_building_types"]
        lane_types[b, :L] = s["token_lane_types"]
        tower_types[b, :L] = s["token_tower_types"]
        ward_types[b, :L] = s["token_ward_types"]
        labels[b, :L] = s["labels"]
        mask[b, :L] = s["label_mask"]
        key_pad[b, :L] = False

    max_T = max(s["anchor_positions"].shape[0] for s in samples)
    max_W = 128

    anchor_positions = torch.zeros(B, max_T, dtype=torch.long)
    frame_features = torch.zeros(B, max_T, 10, FRAME_FEAT_DIM, dtype=torch.float32)
    anchor_macro_features = torch.zeros(B, max_T, MACRO_FEAT_DIM, dtype=torch.float32)
    decision_labels = torch.full((B, max_T, 10), NO_DECISION, dtype=torch.long)
    next_event_type_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_actor_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_target_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_item_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_skill_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_monster_type_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_monster_subtype_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_building_type_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_lane_type_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_tower_type_labels = torch.zeros(B, max_T, dtype=torch.long)
    next_event_ward_type_labels = torch.zeros(B, max_T, dtype=torch.long)
    event_window_raw = torch.zeros(B, max_T, max_W, dtype=torch.long)
    window_mask = torch.zeros(B, max_T, max_W, dtype=torch.float32)

    for bi, s in enumerate(samples):
        T = s["anchor_positions"].shape[0]
        anchor_positions[bi, :T] = torch.from_numpy(s["anchor_positions"])
        frame_features[bi, :T] = torch.from_numpy(s["frame_features"])
        anchor_macro_features[bi, :T] = torch.from_numpy(s["anchor_macro_features"])
        decision_labels[bi, :T] = torch.from_numpy(s["decision_labels"])
        next_event_type_labels[bi, :T] = torch.from_numpy(s["next_event_type_labels"])
        next_event_actor_labels[bi, :T] = torch.from_numpy(s["next_event_actor_labels"])
        next_event_target_labels[bi, :T] = torch.from_numpy(s["next_event_target_labels"])
        next_event_item_labels[bi, :T] = torch.from_numpy(s["next_event_item_labels"])
        next_event_skill_labels[bi, :T] = torch.from_numpy(s["next_event_skill_labels"])
        next_event_monster_type_labels[bi, :T] = torch.from_numpy(s["next_event_monster_type_labels"])
        next_event_monster_subtype_labels[bi, :T] = torch.from_numpy(s["next_event_monster_subtype_labels"])
        next_event_building_type_labels[bi, :T] = torch.from_numpy(s["next_event_building_type_labels"])
        next_event_lane_type_labels[bi, :T] = torch.from_numpy(s["next_event_lane_type_labels"])
        next_event_tower_type_labels[bi, :T] = torch.from_numpy(s["next_event_tower_type_labels"])
        next_event_ward_type_labels[bi, :T] = torch.from_numpy(s["next_event_ward_type_labels"])
        for ti, positions in enumerate(s["window_positions"]):
            positions = positions[:max_W]
            event_window_raw[bi, ti, :len(positions)] = torch.tensor(positions, dtype=torch.long)
            window_mask[bi, ti, :len(positions)] = 1.0

    return {
        "static": torch.stack([s["static"] for s in samples]),
        "players": torch.stack([s["players"] for s in samples]),
        "player_ids": torch.stack([s["player_ids"] for s in samples]),
        "tokens": tokens,
        "token_actors": actors,
        "token_targets": targets,
        "token_timestamps": ts,
        "token_item_ids": item_ids,
        "token_skill_slots": skill_slots,
        "token_monster_types": monster_types,
        "token_monster_subtypes": monster_subtypes,
        "token_building_types": building_types,
        "token_lane_types": lane_types,
        "token_tower_types": tower_types,
        "token_ward_types": ward_types,
        "labels": labels,
        "label_mask": mask,
        "key_pad_mask": key_pad,
        "anchor_positions": anchor_positions,
        "event_window_embeddings_raw": event_window_raw,
        "window_mask": window_mask,
        "frame_features": frame_features,
        "anchor_macro_features": anchor_macro_features,
        "decision_labels": decision_labels,
        "next_event_type_labels": next_event_type_labels,
        "next_event_actor_labels": next_event_actor_labels,
        "next_event_target_labels": next_event_target_labels,
        "next_event_item_labels": next_event_item_labels,
        "next_event_skill_labels": next_event_skill_labels,
        "next_event_monster_type_labels": next_event_monster_type_labels,
        "next_event_monster_subtype_labels": next_event_monster_subtype_labels,
        "next_event_building_type_labels": next_event_building_type_labels,
        "next_event_lane_type_labels": next_event_lane_type_labels,
        "next_event_tower_type_labels": next_event_tower_type_labels,
        "next_event_ward_type_labels": next_event_ward_type_labels,
        "outcome": torch.tensor([s["outcome"] for s in samples], dtype=torch.float32),
    }


def build_puuid_index(match_ids, max_puuids=20000):
    """Assign integer IDs 1..max_puuids-1 to the most-frequent puuids."""
    from collections import Counter

    counts = Counter()
    match_ids = list(match_ids)
    if not match_ids:
        return {}
    conn = get_conn()
    try:
        chunk_size = 900
        for i in range(0, len(match_ids), chunk_size):
            chunk = match_ids[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT DISTINCT match_id, puuid FROM frames "
                f"WHERE match_id IN ({placeholders}) AND participant_slot BETWEEN 1 AND 10",
                chunk,
            ).fetchall()
            for r in rows:
                counts[r["puuid"]] += 1
    finally:
        conn.close()
    ranked = [puuid for puuid, _ in counts.most_common(max_puuids - 1)]
    return {puuid: i + 1 for i, puuid in enumerate(ranked)}


def puuid_ids_for_match(match_id, puuid_index):
    match, _ = get_raw_match(match_id)
    ids = torch.zeros(10, dtype=torch.long)
    if match:
        for i, p in enumerate(match["info"]["participants"][:10]):
            ids[i] = puuid_index.get(p["puuid"], 0)
    return ids
