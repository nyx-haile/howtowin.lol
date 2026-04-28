"""M4 retrieval-check eval harness.

Builds queries from holdout splits, runs k-sweep entropy, prints
per-minute table, runs baseline indexes, writes report. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.

Step 3 / Gate C additions:
- :func:`run_skill_aware_eval` — parallel skill-aware vs unrestricted eval
  over the holdout queries, reporting effective cohort size, widening-stage
  distribution, entropies, and AUC@15.
- :func:`write_retrieval_eval_artifact` — writes ``artifacts/retrieval_eval.json``.
"""
import json
import math
import os
import tempfile

import torch
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games
from model.retrieval import (
    encode_game_keys, encode_game_rank_band, query_index,
    query_index_skill_aware, MID_GAME_MINUTES, SKILL_MIN_EFFECTIVE_K,
    SKILL_OUT_OF_BAND_PENALTY,
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
                           exclude_match_ids, device, mid_minutes,
                           *, log_every: int = 200, log_label: str = "model"):
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
    total = len(holdout_match_ids)

    for i, batch in enumerate(loader, start=1):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        keys, minutes, _ = encode_game_keys(model, batch)
        mask = torch.tensor([int(m.item()) in mid_minute_set for m in minutes],
                            dtype=torch.bool)
        if mask.any():
            keys_list.append(keys[mask].cpu())
            mins_list.append(minutes[mask].cpu())
        if log_every and (i % log_every == 0 or i == total):
            print(f"[m4_eval] {log_label}: encoded {i}/{total} games", flush=True)

    if not keys_list:
        from model.retrieval import KEY_DIM
        return (torch.zeros(0, KEY_DIM), torch.zeros(0, dtype=torch.int64))

    return torch.cat(keys_list, dim=0), torch.cat(mins_list, dim=0)


def _eval_one_index(bundle, queries_raw, query_minutes, *, k_sweep,
                    headline_k, headline_minutes, device, query_batch_size,
                    log_label: str = "model"):
    sweep_means: dict[int, float] = {}
    headline_per_minute: dict[int, float] = {}
    headline_cohort_h = None
    k_values = [int(k) for k in k_sweep]
    if not k_values:
        return {
            "k_sweep": sweep_means,
            "per_minute_at_headline_k": headline_per_minute,
            "headline_cohort_entropies": headline_cohort_h,
        }

    print(
        f"[m4_eval] {log_label}: querying {int(queries_raw.shape[0])} keys "
        f"against corpus_rows={int(bundle.row_blue_win.shape[0])} "
        f"(k_max={max(k_values)})",
        flush=True,
    )
    cohort_idx_wide, _ = query_index(
        bundle, queries_raw, k=max(k_values), device=device,
        batch_size=query_batch_size,
    )
    for k in k_values:
        cohort_idx = cohort_idx_wide[:, :k]
        mean_h = mean_entropy_at_k(cohort_idx, bundle.row_blue_win)
        sweep_means[int(k)] = mean_h
        print(f"[m4_eval] {log_label}: k={k:>3} entropy={mean_h:.3f}", flush=True)
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
        log_label=f"{holdout_label}/model",
    )
    out = {"holdout": holdout_label,
           "n_queries": int(queries_raw.shape[0])}

    out["model"] = _eval_one_index(
        model_bundle, queries_raw, query_minutes,
        k_sweep=k_sweep, headline_k=headline_k,
        headline_minutes=headline_minutes,
        device=device, query_batch_size=query_batch_size,
        log_label=f"{holdout_label}/model",
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
                device, headline_minutes, batch_size=query_batch_size,
                log_label=f"{holdout_label}/static_only",
            )
            out["static_only"] = _eval_one_index(
                static_only_bundle, so_queries[0], so_queries[1],
                k_sweep=k_sweep, headline_k=headline_k,
                headline_minutes=headline_minutes, device=device,
                query_batch_size=query_batch_size,
                log_label=f"{holdout_label}/static_only",
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
                log_label=f"{holdout_label}/frame_features",
            )
    return out


@torch.no_grad()
def _build_static_only_queries(model, holdout_match_ids, puuid_index,
                               exclude_match_ids, device, mid_minutes,
                               batch_size: int = 64,
                               *, log_every: int = 50,
                               log_label: str = "static_only"):
    from model.baselines.static_only_index import encode_static_only_query_keys
    from model.plan_b_model import D_H
    ds = MatchDataset(holdout_match_ids, puuid_index=puuid_index,
                      exclude_match_ids=exclude_match_ids, cache_size=1)
    bs = max(1, int(batch_size))
    loader = DataLoader(
        ds, batch_size=bs, shuffle=False, collate_fn=collate_games,
    )
    keys_list, mins_list = [], []
    mid_set = set(int(m) for m in mid_minutes)
    total_batches = (len(holdout_match_ids) + bs - 1) // bs
    for bi, batch in enumerate(loader, start=1):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        keys, minutes = encode_static_only_query_keys(model, batch)
        mask = torch.tensor([int(m.item()) in mid_set for m in minutes], dtype=torch.bool)
        if mask.any():
            keys_list.append(keys[mask].cpu())
            mins_list.append(minutes[mask])
        if log_every and (bi % log_every == 0 or bi == total_batches):
            print(
                f"[m4_eval] {log_label}: encoded batch {bi}/{total_batches}",
                flush=True,
            )
    if not keys_list:
        return torch.zeros(0, D_H), torch.zeros(0, dtype=torch.int64)
    return torch.cat(keys_list, dim=0), torch.cat(mins_list, dim=0)


def _build_frame_features_queries(holdout_match_ids, mid_minutes):
    from model.baselines.frame_features_index import (
        encode_frame_features_queries, FRAME_BASELINE_DIM,
    )
    keys_list, mins_list = [], []
    mid_set = set(int(m) for m in mid_minutes)
    query_by_mid = encode_frame_features_queries(holdout_match_ids)
    for mid in holdout_match_ids:
        keys, minutes = query_by_mid[mid]
        if keys.shape[0] == 0:
            continue
        mask = torch.tensor([int(m.item()) in mid_set for m in minutes], dtype=torch.bool)
        if mask.any():
            keys_list.append(keys[mask])
            mins_list.append(minutes[mask])
    if not keys_list:
        return torch.zeros(0, FRAME_BASELINE_DIM), torch.zeros(0, dtype=torch.int64)
    return torch.cat(keys_list, dim=0), torch.cat(mins_list, dim=0)


# ---------------------------------------------------------------------------
# Step 3 / Gate C — skill-aware eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def _build_holdout_queries_with_meta(
    model, holdout_match_ids, puuid_index, exclude_match_ids, device, mid_minutes,
):
    """Per-anchor queries plus per-query metadata.

    Returns ``(queries_raw, query_minutes, query_bands, query_blue_win,
    query_match_id)`` — last is a list of len Q.
    """
    from model.retrieval import KEY_DIM

    empty = (
        torch.zeros(0, KEY_DIM),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int8),
        [],
    )
    if not holdout_match_ids:
        return empty

    ds = MatchDataset(holdout_match_ids, puuid_index=puuid_index,
                      exclude_match_ids=exclude_match_ids, cache_size=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)

    keys_list, mins_list, bands_list, bw_list, mid_list = [], [], [], [], []
    mid_set = set(int(m) for m in mid_minutes)

    for i, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        keys, minutes, blue_win = encode_game_keys(model, batch)
        mask = torch.tensor([int(m.item()) in mid_set for m in minutes],
                            dtype=torch.bool)
        if not mask.any():
            continue
        q_band = encode_game_rank_band(batch)
        n_kept = int(mask.sum().item())
        keys_list.append(keys[mask].cpu())
        mins_list.append(minutes[mask].cpu())
        bands_list.append(torch.full((n_kept,), q_band, dtype=torch.int64))
        bw_list.append(torch.full((n_kept,), int(blue_win.item()), dtype=torch.int8))
        mid_list.extend([holdout_match_ids[i]] * n_kept)

    if not keys_list:
        return empty

    return (
        torch.cat(keys_list, dim=0),
        torch.cat(mins_list, dim=0),
        torch.cat(bands_list, dim=0),
        torch.cat(bw_list, dim=0),
        mid_list,
    )


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro binary AUC via sklearn. Returns nan if <2 classes present."""
    from sklearn.metrics import roc_auc_score
    y = labels.cpu().numpy()
    s = scores.cpu().numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


@torch.no_grad()
def run_skill_aware_eval(
    *,
    model,
    model_bundle,
    holdout_match_ids: list[str],
    puuid_index: dict,
    exclude_match_ids: set[str],
    k: int = 64,
    device: str = "cpu",
    query_batch_size: int = 256,
    skill_batch_size: int = 64,
    min_effective_k: int = SKILL_MIN_EFFECTIVE_K,
    out_of_band_penalty: float = SKILL_OUT_OF_BAND_PENALTY,
    auc_minute: int = 15,
) -> dict:
    """Dual-report eval: unrestricted vs. skill-aware retrieval on the holdout.

    Returns a dict shaped for ``artifacts/retrieval_eval.json``.
    """
    queries_raw, q_min, q_band, q_bw, q_mid = _build_holdout_queries_with_meta(
        model, holdout_match_ids, puuid_index, exclude_match_ids,
        device, MID_GAME_MINUTES,
    )

    Q = int(queries_raw.shape[0])
    out: dict = {
        "n_queries": Q,
        "n_holdout_games": len(holdout_match_ids),
        "k": int(k),
    }

    if Q == 0:
        out["gate_c_pass"] = False
        out["median_effective_k"] = 0.0
        out["skill_aware_entropy_at_64"] = float("nan")
        out["unrestricted_entropy_at_64"] = float("nan")
        out["auc_at_15"] = float("nan")
        out["widening_stage_counts"] = {"same": 0, "adjacent": 0, "unrestricted": 0}
        return out

    # Unrestricted retrieval (classic path).
    u_idx, _ = query_index(
        model_bundle, queries_raw, k=k, device=device,
        batch_size=query_batch_size,
    )
    u_entropy = mean_entropy_at_k(u_idx, model_bundle.row_blue_win)
    u_cohort_winrate = model_bundle.row_blue_win[u_idx].float().mean(dim=1)

    # Skill-aware retrieval.
    s_idx, _, info = query_index_skill_aware(
        model_bundle, queries_raw, k=k,
        query_rank_bands=q_band.long(),
        device=device,
        batch_size=skill_batch_size,
        min_effective_k=min_effective_k,
        out_of_band_penalty=out_of_band_penalty,
    )
    s_entropy = mean_entropy_at_k(s_idx, model_bundle.row_blue_win)
    s_cohort_winrate = model_bundle.row_blue_win[s_idx].float().mean(dim=1)

    # AUC @ auc_minute — cohort-mean blue_win as the soft predictor,
    # query_blue_win as the true label. We pick the anchor closest to
    # auc_minute per *holdout game* to avoid over-counting long games.
    auc_mask = (q_min == int(auc_minute))
    if auc_mask.sum().item() < 2:
        auc_unrestricted = float("nan")
        auc_skill = float("nan")
    else:
        auc_unrestricted = _roc_auc(u_cohort_winrate[auc_mask], q_bw[auc_mask])
        auc_skill = _roc_auc(s_cohort_winrate[auc_mask], q_bw[auc_mask])

    # Widening-stage distribution.
    stages = info["widening_stage_per_query"]
    stage_counts = {
        "same": int((stages == 0).sum().item()),
        "adjacent": int((stages == 1).sum().item()),
        "unrestricted": int((stages == 2).sum().item()),
    }

    # Effective cohort size reported as the SAME-band eligibility per query
    # (rows that pass the strict in-band filter). Median is the headline gate.
    eff_k = info["effective_k_per_query"].float()
    median_eff_k = float(eff_k.median().item())
    mean_eff_k = float(eff_k.mean().item())
    in_band_frac = info["in_band_fraction_per_query"]
    # Drop nan entries (unranked queries) before aggregating.
    in_band_clean = in_band_frac[~torch.isnan(in_band_frac)]
    mean_in_band = (
        float(in_band_clean.mean().item())
        if in_band_clean.numel() else float("nan")
    )

    gate_c_pass = bool(
        (median_eff_k >= float(min_effective_k))
        and (not math.isnan(auc_skill))
        and (auc_skill >= auc_unrestricted - 0.02 if not math.isnan(auc_unrestricted) else True)
        and (not math.isnan(s_entropy))
        and (not math.isnan(u_entropy))
        and (s_entropy <= u_entropy + 0.03)
    )

    out.update({
        "gate_c_pass": gate_c_pass,
        "median_effective_k": median_eff_k,
        "mean_effective_k": mean_eff_k,
        "skill_aware_entropy_at_64": float(s_entropy),
        "unrestricted_entropy_at_64": float(u_entropy),
        "auc_at_15": float(auc_skill),
        "auc_at_15_unrestricted": float(auc_unrestricted),
        "mean_in_band_fraction": mean_in_band,
        "widening_stage_counts": stage_counts,
        "auc_n_queries": int(auc_mask.sum().item()),
        "config": {
            "min_effective_k": int(min_effective_k),
            "out_of_band_penalty": float(out_of_band_penalty),
            "auc_minute": int(auc_minute),
        },
    })
    return out


def write_retrieval_eval_artifact(path: str, metrics: dict) -> None:
    """Atomic JSON write for the Gate C handoff artifact."""
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dirn or ".", prefix=".retrieval_eval.", suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(metrics, tmp, indent=2, sort_keys=True)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
