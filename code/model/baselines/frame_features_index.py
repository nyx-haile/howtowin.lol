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
