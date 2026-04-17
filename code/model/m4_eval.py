"""M4 retrieval-check eval harness.

Builds queries from holdout splits, runs k-sweep entropy, prints
per-minute table, runs baseline indexes, writes report. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import math
import torch
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games
from model.retrieval import (
    encode_game_keys, query_index, MID_GAME_MINUTES,
)

_LOG2 = math.log(2.0)


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-element binary entropy in bits. Returns 0 at p in {0, 1}."""
    # torch.xlogy(x, y) = x * log(y), with xlogy(0, 0) = 0 by convention,
    # so this is numerically correct at both endpoints without clamping.
    h_nats = -(torch.xlogy(p, p) + torch.xlogy(1.0 - p, 1.0 - p))
    return h_nats / _LOG2


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
