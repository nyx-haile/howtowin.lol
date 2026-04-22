"""Static-only retrieval baseline: kNN on a pooled tokenized static-context key.

The key is a per-game 512-dim summary derived only from the tokenized static
context stream, copied per anchor for shape parity with the headline index.

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
    static_summary = model.encode_static_summary(batch["static"])  # (1, D_H)
    return static_summary[0].detach().cpu()


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
    original_device = next(model.parameters()).device
    model.to(device)
    try:
        return _build_static_only_inner(
            model=model, train_match_ids=train_match_ids,
            exclude_match_ids=exclude_match_ids, puuid_index=puuid_index,
            device=device, log_every=log_every,
        )
    finally:
        model.to(original_device)


def _build_static_only_inner(
    *, model, train_match_ids, exclude_match_ids, puuid_index, device, log_every,
) -> IndexBundle:
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


@torch.no_grad()
def encode_static_only_query_key(model, batch) -> torch.Tensor:
    """(T, D_H) — copy the per-game static key per anchor."""
    key = _encode_game_static_key(model, batch)            # (D_H,)
    T = batch["anchor_positions"].size(1)
    return key.unsqueeze(0).expand(T, -1).clone()
