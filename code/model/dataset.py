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
