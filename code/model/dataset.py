"""PyTorch Dataset wrapping tokenizer + features + labels.

Each sample is one game. Labels are per-anchor: the distribution of
meaningful event types fired in the minute following that anchor
(multi-hot). Training target at each anchor is this multi-hot vector;
top-5 eval asks whether the top-5 predicted classes cover the truth.

Plan A uses multi-hot targets to match "top-5 of next-minute events".
"""
import os
from collections import OrderedDict
import numpy as np
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

# Decision types for next-decision head.
DECISION_MAP = {
    EVENT_TYPE_TO_ID.get("ITEM_PURCHASED", -1): 0,
    EVENT_TYPE_TO_ID.get("SKILL_LEVEL_UP", -1): 1,
    EVENT_TYPE_TO_ID.get("WARD_PLACED", -1): 2,
    EVENT_TYPE_TO_ID.get("RECALL", -1): 3,
    EVENT_TYPE_TO_ID.get("ENGAGE", -1): 4,
    EVENT_TYPE_TO_ID.get("DISENGAGE", -1): 5,
}
DECISION_MAP.pop(-1, None)  # remove sentinel if any key wasn't found
NO_DECISION = 6
FRAME_FEAT_DIM = 6

# Per-feature normalization scales for frame features so the model sees O(1) values.
# Order: total_gold, xp, level, pos_x, pos_y, cs
FRAME_FEAT_SCALE = np.array([5000.0, 5000.0, 18.0, 15000.0, 15000.0, 200.0], dtype=np.float32)


SPLIT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'splits')


def load_split(name):
    """name in {'holdout', 'train', 'cold'}."""
    holdout_path = os.path.join(SPLIT_DIR, 'plan_a_holdout.txt')
    with open(holdout_path) as f:
        holdout = [line.strip() for line in f if line.strip()]

    if name == 'cold':
        from model.cold_holdout import load_player_cold_holdout
        return sorted(load_player_cold_holdout())

    if name == 'holdout':
        return holdout

    # 'train' — subtract both game-cold and player-cold holdouts.
    conn = get_conn()
    try:
        all_ids = [r["match_id"] for r in
                   conn.execute("SELECT match_id FROM games").fetchall()]
    finally:
        conn.close()
    hset = set(holdout)
    from model.cold_holdout import load_player_cold_holdout
    try:
        player_cold = load_player_cold_holdout()
    except FileNotFoundError:
        player_cold = set()
    return [m for m in all_ids if m not in hset and m not in player_cold]


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
    def __init__(self, match_ids, puuid_index=None, exclude_match_ids=None,
                 cache_size=8000):
        self.match_ids = list(match_ids)
        self.puuid_index = puuid_index or {}
        self.exclude_match_ids = set(exclude_match_ids) if exclude_match_ids else set()
        self._cache = OrderedDict()  # LRU cache; capped to avoid OOM with large corpora
        self._cache_max = min(len(self.match_ids), cache_size)
        self._player_feat = self._preload_player_features()

    def _preload_player_features(self):
        """Precompute player feature vectors for every puuid in the split.

        Without this, every cache-miss game load triggers 50 SQLite queries
        (5 per player × 10 players). Precomputing once at init amortises to
        ~5 queries per unique puuid.
        """
        if not self.match_ids:
            return {}
        conn = get_conn()
        try:
            puuids = set()
            # SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999; chunk the IN list.
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
        token_ts = torch.tensor([t.timestamp_ms for t in tokens], dtype=torch.float32)

        static = torch.tensor(patch_vector_for_match(mid), dtype=torch.float32)

        match, _ = get_raw_match(mid)
        players = torch.zeros(10, PLAYER_FEATURE_DIM, dtype=torch.float32)
        player_ids = torch.zeros(10, dtype=torch.long)
        if match:
            for i_p, p in enumerate(match["info"]["participants"][:10]):
                feat = self._player_feat.get(p["puuid"])
                if feat is not None:
                    players[i_p] = torch.from_numpy(feat)
                player_ids[i_p] = self.puuid_index.get(p["puuid"], 0)

        # --- Plan B anchor windowing ---
        anchor_idx = [j for j, tok in enumerate(tokens) if tok.type_id == ANCHOR_TOKEN]

        # Frame features per anchor — one query for the whole game, then pivot.
        conn = get_conn()
        try:
            all_frame_rows = conn.execute(
                "SELECT timestamp_ms, participant_slot, total_gold, xp, level, pos_x, pos_y, cs "
                "FROM frames WHERE match_id = ? AND participant_slot BETWEEN 1 AND 10",
                (mid,),
            ).fetchall()
            ts_to_slots = {}
            for r in all_frame_rows:
                ts_to_slots.setdefault(r["timestamp_ms"], {})[r["participant_slot"]] = r

            frame_feats = np.zeros((len(anchor_idx), 10, FRAME_FEAT_DIM), dtype=np.float32)
            for ai, a_i in enumerate(anchor_idx):
                slot_map = ts_to_slots.get(tokens[a_i].timestamp_ms, {})
                for slot, r in slot_map.items():
                    frame_feats[ai, slot - 1] = [
                        float(r["total_gold"] or 0), float(r["xp"] or 0),
                        float(r["level"] or 0), float(r["pos_x"] or 0),
                        float(r["pos_y"] or 0), float(r["cs"] or 0),
                    ]

            # Normalize frame features to O(1) range.
            frame_feats /= FRAME_FEAT_SCALE[np.newaxis, np.newaxis, :]

            # Outcome: blue team (100) win = 1, red (200) win = 0.
            game_row = conn.execute(
                "SELECT winning_team FROM games WHERE match_id = ?", (mid,)
            ).fetchone()
            outcome = 1 if game_row and game_row["winning_team"] == 100 else 0
        finally:
            conn.close()

        # Decision labels per anchor per participant.
        decision_labels = np.full((len(anchor_idx), 10), NO_DECISION, dtype=np.int64)
        for ti, a_i in enumerate(anchor_idx):
            nxt = anchor_idx[ti + 1] if ti + 1 < len(anchor_idx) else len(tokens)
            for tok in tokens[a_i + 1:nxt]:
                if tok.type_id in DECISION_MAP and 1 <= tok.actor_slot <= 10:
                    decision_labels[ti, tok.actor_slot - 1] = DECISION_MAP[tok.type_id]

        # Event-window token positions (ragged).
        window_positions = []
        for ti, a_i in enumerate(anchor_idx):
            nxt = anchor_idx[ti + 1] if ti + 1 < len(anchor_idx) else len(tokens)
            window_positions.append(list(range(a_i + 1, nxt)))

        return {
            "static": static,
            "players": players,
            "player_ids": player_ids,
            "tokens": token_ids,
            "token_actors": token_actors,
            "token_timestamps": token_ts,
            "labels": labels,
            "label_mask": label_mask,
            "anchor_positions": np.array(anchor_idx, dtype=np.int64),
            "frame_features": frame_feats,
            "decision_labels": decision_labels,
            "window_positions": window_positions,
            "outcome": outcome,
        }


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

    # Anchor-windowing tensors (Plan B).
    max_T = max(s["anchor_positions"].shape[0] for s in samples)
    max_W = 128  # cap events per window; truncate if more

    anchor_positions = torch.zeros(B, max_T, dtype=torch.long)
    frame_features = torch.zeros(B, max_T, 10, FRAME_FEAT_DIM, dtype=torch.float32)
    decision_labels = torch.full((B, max_T, 10), NO_DECISION, dtype=torch.long)
    event_window_raw = torch.zeros(B, max_T, max_W, dtype=torch.long)
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
        "anchor_positions": anchor_positions,
        "event_window_embeddings_raw": event_window_raw,
        "window_mask": window_mask,
        "frame_features": frame_features,
        "decision_labels": decision_labels,
        "outcome": torch.tensor([s["outcome"] for s in samples], dtype=torch.float32),
    }


def build_puuid_index(match_ids, max_puuids=20000):
    """Assign integer IDs 1..max_puuids-1 to the most-frequent puuids in the
    given match_ids. Returns dict puuid -> id. Unknown puuids resolve to 0.
    """
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
