"""Rank-use diagnostics (Step 2 / Gate B of the skill-causal plan).

Probes a trained Plan B model for whether rank information is
*under-used* (present in features but not in dynamics) or *over-leaked*
(dominating the latent state and confounding retrieval). The module is
read-only with respect to ``retrieval.py`` — the retrieval-key assembly
is mirrored inline to avoid touching that file.

Outputs ``artifacts/rank_diagnosis.json`` with the schema documented in
``docs/superpowers/plans/2026-04-24-skill-causal-team-map.md``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from db import get_conn
from model.dataset import collate_games

# Coarse bands — see plan's "Retrieval / rank-band contract" section.
# Tier ordinal (RANK_TIER_ORDER in player_features.py) -> coarse band id.
# 0 (unranked) maps to -1 and is filtered out before probe fit.
TIER_TO_BAND: dict[int, int] = {
    0: -1,
    1: 0, 2: 0, 3: 0,           # Iron, Bronze, Silver
    4: 1, 5: 1,                 # Gold, Platinum
    6: 2, 7: 2,                 # Emerald, Diamond
    8: 3, 9: 3, 10: 3,          # Master, Grandmaster, Challenger
}
BAND_NAMES = ["iron_silver", "gold_platinum", "emerald_diamond", "master_plus"]
N_BANDS = 4

PROBE_LAYERS = ("player_emb", "h_t", "post_mu", "prior_mu", "retrieval_key")

BAND_TIER_NAMES: dict[int, tuple[str, ...]] = {
    0: ("IRON", "BRONZE", "SILVER"),
    1: ("GOLD", "PLATINUM"),
    2: ("EMERALD", "DIAMOND"),
    3: ("MASTER", "GRANDMASTER", "CHALLENGER"),
}


def select_band_stratified_match_ids(
    candidate_match_ids: list[str],
    per_band_cap: int,
    *,
    seed: int = 0,
) -> tuple[list[str], dict[str, int]]:
    """Return a balanced sample: up to ``per_band_cap`` games per coarse band.

    Queries the DB for every candidate match to find which coarse bands have
    at least one ranked player in the lobby. The corpus is severely
    imbalanced (master_plus dominates ~476:1 over iron_silver), so uniform
    sampling cannot produce a multi-class probe. This sampler enforces
    stratification.

    Returns ``(match_ids, counts_by_band_name)``. A match is counted once
    per band it satisfies (a game with both a gold and a diamond player
    appears in both gold_platinum and emerald_diamond eligibility sets,
    sampled independently per band up to the cap).
    """
    if not candidate_match_ids:
        return [], {name: 0 for name in BAND_NAMES}

    import numpy as np

    rng = np.random.default_rng(seed)
    conn = get_conn()
    eligible: dict[int, list[str]] = {b: [] for b in BAND_TIER_NAMES}
    try:
        cand_set = set(candidate_match_ids)
        for band_id, tiers in BAND_TIER_NAMES.items():
            ph = ",".join("?" * len(tiers))
            rows = conn.execute(
                f"SELECT DISTINCT pms.match_id "
                f"FROM player_match_stats pms "
                f"JOIN players p ON p.puuid = pms.puuid "
                f"WHERE p.rank_tier IN ({ph})",
                list(tiers),
            ).fetchall()
            band_matches = [r["match_id"] for r in rows if r["match_id"] in cand_set]
            eligible[band_id] = band_matches
    finally:
        conn.close()

    picked: set[str] = set()
    counts: dict[str, int] = {name: 0 for name in BAND_NAMES}
    # Sample smallest bands first so they're not crowded out by overlap with master+.
    band_order = sorted(eligible.keys(), key=lambda b: len(eligible[b]))
    for band_id in band_order:
        remaining = [m for m in eligible[band_id] if m not in picked]
        take = min(per_band_cap, len(remaining))
        if take:
            idx = rng.choice(len(remaining), size=take, replace=False)
            chosen = [remaining[i] for i in idx]
            picked.update(chosen)
            counts[BAND_NAMES[band_id]] = take
        else:
            counts[BAND_NAMES[band_id]] = 0
    return sorted(picked), counts


def _tier_to_band_tensor(tiers: torch.Tensor) -> torch.Tensor:
    """Map tier ordinal (0..10) to coarse band id (0..3) or -1 (unranked)."""
    lut = torch.tensor(
        [-1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
        dtype=torch.long,
        device=tiers.device,
    )
    t = tiers.round().long().clamp(min=0, max=10)
    band = lut[t]
    band[tiers <= 0] = -1
    return band


MIN_RANKED_PLAYERS_FOR_GAME_BAND = 1


def _game_band(players: torch.Tensor) -> torch.Tensor:
    """Per-game rank band: mode of ranked players' coarse bands.

    ``players``: ``(B, 10, PLAYER_FEATURE_DIM)`` — col 0 is ``rank_tier_ordinal``.
    Returns ``(B,)`` long; value in ``{0..3}`` if at least
    ``MIN_RANKED_PLAYERS_FOR_GAME_BAND`` players are ranked, else ``-1``.

    Rationale: the 51k corpus has ~19% per-puuid rank coverage, and in this
    corpus "unranked" usually means "rank backfill pending" rather than the
    player being truly unranked. Matchmaking binds a lobby tightly, so even
    one ranked player's band is a faithful proxy for the lobby's true band.
    """
    player_bands = _player_band(players)   # (B, 10) long, -1 for unranked
    B = player_bands.shape[0]
    out = torch.full((B,), -1, dtype=torch.long, device=player_bands.device)
    for b in range(B):
        ranked = player_bands[b][player_bands[b] >= 0]
        if ranked.numel() < MIN_RANKED_PLAYERS_FOR_GAME_BAND:
            continue
        vals, counts = torch.unique(ranked, return_counts=True)
        out[b] = vals[counts.argmax()]
    return out


def _player_band(players: torch.Tensor) -> torch.Tensor:
    """Per-player band. Returns ``(B, 10)`` long, values in ``{0..3}`` or ``-1``."""
    tiers = players[..., 0]
    return _tier_to_band_tensor(tiers)


def _balance_sample(
    features: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Subsample at most ``max_per_class`` per class for a balanced probe fit."""
    shards_x: list[np.ndarray] = []
    shards_y: list[np.ndarray] = []
    for c in sorted(np.unique(labels).tolist()):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        shards_x.append(features[idx])
        shards_y.append(labels[idx])
    if not shards_x:
        return None, None
    return np.concatenate(shards_x, axis=0), np.concatenate(shards_y, axis=0)


def fit_shallow_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_per_class_train: int = 2000,
    max_per_class_test: int = 500,
    seed: int = 0,
) -> float:
    """Multinomial logistic probe on ``(features, labels)``. Returns macro OvR AUC.

    Returns ``nan`` if the probe cannot be fit (too few samples / classes).
    """
    rng = np.random.default_rng(seed)
    mask = labels >= 0
    X = features[mask]
    y = labels[mask]
    if X.shape[0] < 100 or len(np.unique(y)) < 2:
        return float("nan")
    perm = rng.permutation(X.shape[0])
    split = int(X.shape[0] * 0.7)
    tr_idx, te_idx = perm[:split], perm[split:]
    X_tr, y_tr = _balance_sample(X[tr_idx], y[tr_idx], max_per_class_train, rng)
    X_te, y_te = _balance_sample(X[te_idx], y[te_idx], max_per_class_test, rng)
    if X_tr is None or X_te is None:
        return float("nan")
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)
    return float(
        roc_auc_score(
            y_te, proba,
            multi_class="ovr", average="macro",
            labels=clf.classes_,
        )
    )


@dataclass(frozen=True)
class DiagnoseConfig:
    device: str
    max_anchors_per_layer: int
    sample_games: int
    swap_sample: int
    probe_seed: int = 0


def _to_device(batch: Mapping, device: str) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def collect_activations_and_collapse(
    model, dataset, *, config: DiagnoseConfig,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Single-pass collector: probe activations AND collapse/KL monitors.

    Returns ``(activations, collapse)`` where:
      - ``activations[layer] = (X, y)`` for each PROBE_LAYERS member;
      - ``collapse`` has per-dim KL / active-units / post_mu variance.

    Merged to avoid a second forward pass over the same games.
    """
    loader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_games, shuffle=False, num_workers=0,
    )
    per_layer: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in PROBE_LAYERS
    }
    counts: dict[str, int] = {name: 0 for name in PROBE_LAYERS}
    sum_kl_per_dim: torch.Tensor | None = None
    sum_anchors = 0
    post_mu_shards: list[np.ndarray] = []
    n_games = 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            if n_games >= config.sample_games:
                break
            n_games += 1
            batch = _to_device(batch, config.device)
            players = batch["players"]
            p_bands = _player_band(players).cpu().numpy()    # (1, 10)
            g_band = _game_band(players).cpu().numpy()       # (1,)

            # Capture player_emb by re-invoking the encoder (same call
            # PlanBModel.forward uses internally — deterministic).
            player_emb = model.player_enc(players, batch["player_ids"])  # (1, 10, D_MODEL)
            out = model(batch)

            if counts["player_emb"] < config.max_anchors_per_layer:
                pe = player_emb[0].detach().cpu().numpy()
                per_layer["player_emb"].append((pe, p_bands[0]))
                counts["player_emb"] += pe.shape[0]

            am_bool = out["anchor_mask"].bool()
            am_np = am_bool[0].cpu().numpy().astype(bool)     # (T,)

            # Collapse / KL accumulator (uses all valid anchors even if the
            # per-layer activation caps are already full).
            pm = out["post_mu"]
            plv = out["post_logvar"]
            rm = out["prior_mu"]
            rlv = out["prior_logvar"]
            var_q = plv.exp()
            var_p = rlv.exp()
            kl = 0.5 * (rlv - plv + (var_q + (pm - rm) ** 2) / var_p - 1.0)  # (B, T, D_Z)
            kl_flat = kl[am_bool]
            if kl_flat.numel() > 0:
                if sum_kl_per_dim is None:
                    sum_kl_per_dim = kl_flat.sum(dim=0)
                else:
                    sum_kl_per_dim = sum_kl_per_dim + kl_flat.sum(dim=0)
                sum_anchors += kl_flat.shape[0]
                post_mu_shards.append(pm[am_bool].detach().cpu().numpy())

            if not am_np.any():
                continue

            h = out["h"][0].detach().cpu().numpy()
            post_mu = out["post_mu"][0].detach().cpu().numpy()
            prior_mu = out["prior_mu"][0].detach().cpu().numpy()
            # Retrieval key = [h || post_mu] — mirrors retrieval.encode_game_keys (read-only).
            rk = np.concatenate([h, post_mu], axis=-1)

            n_v = int(am_np.sum())
            label_vec = np.full((n_v,), int(g_band[0]), dtype=np.int64)
            for name, feat in (
                ("h_t", h),
                ("post_mu", post_mu),
                ("prior_mu", prior_mu),
                ("retrieval_key", rk),
            ):
                if counts[name] < config.max_anchors_per_layer:
                    per_layer[name].append((feat[am_np], label_vec))
                    counts[name] += n_v
    finally:
        model.train(was_training)

    activations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, shards in per_layer.items():
        if not shards:
            activations[name] = (np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64))
            continue
        X = np.concatenate([s[0] for s in shards], axis=0)
        y = np.concatenate([s[1] for s in shards], axis=0)
        activations[name] = (X.astype(np.float32), y.astype(np.int64))

    if sum_anchors == 0 or sum_kl_per_dim is None:
        collapse = {
            "per_dim_kl": [],
            "active_units": 0,
            "post_mu_variance_per_dim": [],
            "post_mu_variance_mean": 0.0,
            "n_anchors": 0,
        }
    else:
        mean_kl_per_dim = (sum_kl_per_dim / float(sum_anchors)).detach().cpu().numpy()
        active_units = int((mean_kl_per_dim > 0.01).sum())
        post_mu = np.concatenate(post_mu_shards, axis=0) if post_mu_shards else np.zeros((0, 0))
        post_mu_var = post_mu.var(axis=0) if post_mu.size else np.zeros(0)
        collapse = {
            "per_dim_kl": [float(v) for v in mean_kl_per_dim.tolist()],
            "active_units": active_units,
            "post_mu_variance_per_dim": [float(v) for v in post_mu_var.tolist()],
            "post_mu_variance_mean": float(post_mu_var.mean()) if post_mu_var.size else 0.0,
            "n_anchors": int(sum_anchors),
        }
    return activations, collapse


# Back-compat shims used by existing tests / callers that want either piece.
@torch.no_grad()
def collect_layer_activations(model, dataset, *, config: DiagnoseConfig):
    activations, _ = collect_activations_and_collapse(model, dataset, config=config)
    return activations


def _score_at_minute(batch, scores: torch.Tensor, anchor_mask: torch.Tensor, target: int) -> float:
    """Predicted P(blue_win) at the first valid anchor at or after ``target`` minute."""
    pos = batch["anchor_positions"][0]
    ts = batch["token_timestamps"][0]
    minutes = (ts.gather(0, pos.long()).float() / 60000.0).round().long()
    valid = anchor_mask.bool()
    if not valid.any():
        return float("nan")
    candidates = torch.where(valid & (minutes >= target))[0]
    if candidates.numel() == 0:
        candidates = torch.where(valid)[0]
    if candidates.numel() == 0:
        return float("nan")
    return float(scores[candidates[0]].item())


@torch.no_grad()
def run_swap_and_ablation(
    model, dataset, *, config: DiagnoseConfig,
) -> dict[str, float]:
    """Rank swap + rank-only / no-rank ablations.

    ``swap_delta`` is the mean absolute change in predicted P(blue_win) at
    the first anchor at or after minute 15 when the 10 players' rank/LP
    features are replaced with those of the previous game.
    """
    loader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_games, shuffle=False, num_workers=0,
    )
    was_training = model.training
    model.eval()

    y_true: list[float] = []
    p_base: list[float] = []
    p_swap: list[float] = []
    p_no_rank: list[float] = []
    p_rank_only: list[float] = []
    prev_rank_lp: torch.Tensor | None = None
    n = 0

    try:
        for batch in loader:
            if n >= config.swap_sample:
                break
            n += 1
            batch = _to_device(batch, config.device)
            y_true.append(float(batch["outcome"][0].item()))

            out_base = model(batch)
            scores_base = torch.sigmoid(out_base["outcome_logits"][0])
            p_base.append(_score_at_minute(batch, scores_base, out_base["anchor_mask"][0], 15))

            players = batch["players"]

            if prev_rank_lp is not None and prev_rank_lp.shape == players[..., :2].shape:
                players_swap = players.clone()
                players_swap[..., :2] = prev_rank_lp
                out_swap = model({**batch, "players": players_swap})
                scores_swap = torch.sigmoid(out_swap["outcome_logits"][0])
                p_swap.append(_score_at_minute(batch, scores_swap, out_swap["anchor_mask"][0], 15))
            else:
                p_swap.append(float("nan"))

            players_no_rank = players.clone()
            players_no_rank[..., :2] = 0.0
            out_nr = model({**batch, "players": players_no_rank})
            scores_nr = torch.sigmoid(out_nr["outcome_logits"][0])
            p_no_rank.append(_score_at_minute(batch, scores_nr, out_nr["anchor_mask"][0], 15))

            players_rank_only = players.clone()
            players_rank_only[..., 2:] = 0.0
            out_ro = model({**batch, "players": players_rank_only})
            scores_ro = torch.sigmoid(out_ro["outcome_logits"][0])
            p_rank_only.append(_score_at_minute(batch, scores_ro, out_ro["anchor_mask"][0], 15))

            prev_rank_lp = players[..., :2].detach().clone()
    finally:
        model.train(was_training)

    y_arr = np.array(y_true, dtype=np.float64)

    def _auc(probs: list[float]) -> float:
        p_arr = np.array(probs, dtype=np.float64)
        ok = ~np.isnan(p_arr)
        if ok.sum() < 2 or len(set(y_arr[ok].tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(y_arr[ok], p_arr[ok]))

    base_arr = np.array(p_base, dtype=np.float64)
    swap_arr = np.array(p_swap, dtype=np.float64)
    ok = ~(np.isnan(base_arr) | np.isnan(swap_arr))
    swap_delta = float(np.mean(np.abs(base_arr[ok] - swap_arr[ok]))) if ok.any() else float("nan")

    return {
        "baseline_auc_m15": _auc(p_base),
        "swap_auc_m15": _auc(p_swap),
        "no_rank_auc_m15": _auc(p_no_rank),
        "rank_only_auc_m15": _auc(p_rank_only),
        "swap_delta": swap_delta,
        "n_games_swap": int(ok.sum()),
    }


@torch.no_grad()
def run_collapse_monitors(model, dataset, *, config: DiagnoseConfig) -> dict[str, Any]:
    """Back-compat wrapper: runs the merged collector and returns only collapse stats."""
    _, collapse = collect_activations_and_collapse(model, dataset, config=config)
    return collapse


def classify_rank_use(probe_auc: Mapping[str, float], swap_delta: float) -> str:
    """Map probe AUCs + swap_delta to ``under_use``/``over_leak``/``mixed_or_unclear``.

    Thresholds are documented in the returned artifact ``details`` for auditability.

    - ``under_use``: rank is linearly readable in the encoder output (``player_emb``)
      but not in the dynamics (``h_t``). Criterion: ``auc(player_emb) >= 0.70`` and
      ``auc(h_t) < 0.60``.
    - ``over_leak``: dynamics and latent both encode rank strongly *and* swapping
      rank meaningfully changes outcome predictions. Criterion: ``auc(h_t) >= 0.75``
      and ``auc(post_mu) >= 0.70`` and ``swap_delta >= 0.05``.
    """
    auc_emb = float(probe_auc.get("player_emb", float("nan")))
    auc_h = float(probe_auc.get("h_t", float("nan")))
    auc_pmu = float(probe_auc.get("post_mu", float("nan")))

    if (
        not np.isnan(auc_emb) and not np.isnan(auc_h)
        and auc_emb >= 0.70 and auc_h < 0.60
    ):
        return "under_use"

    if (
        not np.isnan(auc_h) and not np.isnan(auc_pmu)
        and auc_h >= 0.75 and auc_pmu >= 0.70
        and not np.isnan(swap_delta) and swap_delta >= 0.05
    ):
        return "over_leak"

    return "mixed_or_unclear"


def write_rank_diagnosis_artifact(path: str, metrics: Mapping) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(dict(metrics), f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[rank-diagnose] wrote {path}", flush=True)


def run_gate_b_diagnosis(
    *,
    checkpoint_path: str,
    split: str = "holdout",
    sample_games: int = 500,
    swap_sample: int = 500,
    max_anchors_per_layer: int = 30000,
    artifact_path: str = "artifacts/rank_diagnosis.json",
    device: str | None = None,
    band_stratified: bool = True,
    per_band_cap: int = 250,
) -> dict:
    """End-to-end Gate B diagnosis. Writes ``artifact_path`` and returns metrics."""
    from model.plan_b_model import PlanBModel
    from model.dataset import MatchDataset, build_puuid_index, load_split

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rank-diagnose] device={device} checkpoint={checkpoint_path}", flush=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)

    split_ids = load_split(split)
    val_ids = load_split("holdout")
    cold_ids = load_split("cold")
    train_ids = load_split("train")
    exclude = set(val_ids) | set(cold_ids)
    puuid_index = build_puuid_index(train_ids, max_puuids=max_puuids)

    model = PlanBModel(max_puuids=max_puuids).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if band_stratified:
        subset_ids, stratification_counts = select_band_stratified_match_ids(
            list(split_ids), per_band_cap=per_band_cap, seed=0,
        )
        if len(subset_ids) > sample_games:
            # Trim deterministically to the requested cap, preserving stratification.
            subset_ids = subset_ids[:sample_games]
        print(
            f"[rank-diagnose] band-stratified sample: total={len(subset_ids)} "
            f"per_band_cap={per_band_cap} counts={stratification_counts}",
            flush=True,
        )
    else:
        subset_ids = list(split_ids[:sample_games])
        stratification_counts = None
    n_use = len(subset_ids)
    ds = MatchDataset(subset_ids, puuid_index, exclude_match_ids=exclude)

    config = DiagnoseConfig(
        device=device,
        max_anchors_per_layer=max_anchors_per_layer,
        sample_games=n_use,
        swap_sample=min(swap_sample, n_use),
    )

    print(f"[rank-diagnose] collect activations + collapse: split={split} games={n_use}", flush=True)
    activations, collapse = collect_activations_and_collapse(model, ds, config=config)
    probe_auc: dict[str, float] = {}
    for name in PROBE_LAYERS:
        X, y = activations[name]
        if X.shape[0] == 0:
            probe_auc[name] = float("nan")
            print(f"[rank-diagnose] {name:>14}: empty", flush=True)
            continue
        probe_auc[name] = fit_shallow_probe(X, y, seed=config.probe_seed)
        print(f"[rank-diagnose] {name:>14}: auc={probe_auc[name]:.3f}  n={X.shape[0]}", flush=True)
    print(
        f"[rank-diagnose] active_units={collapse['active_units']} / "
        f"{len(collapse['per_dim_kl'])}  mean_post_mu_var={collapse['post_mu_variance_mean']:.3f}",
        flush=True,
    )

    print(f"[rank-diagnose] swap + ablation: games={config.swap_sample}", flush=True)
    swap_abl = run_swap_and_ablation(model, ds, config=config)
    print(
        f"[rank-diagnose] swap_delta={swap_abl['swap_delta']:.3f} "
        f"baseline_m15={swap_abl['baseline_auc_m15']:.3f} "
        f"swap_m15={swap_abl['swap_auc_m15']:.3f} "
        f"no_rank_m15={swap_abl['no_rank_auc_m15']:.3f} "
        f"rank_only_m15={swap_abl['rank_only_auc_m15']:.3f}",
        flush=True,
    )

    classification = classify_rank_use(probe_auc, swap_abl["swap_delta"])
    gate_b_pass = classification in ("under_use", "over_leak", "mixed_or_unclear")

    metrics = {
        "classification": classification,
        "probe_auc_by_layer": probe_auc,
        "swap_delta": swap_abl["swap_delta"],
        "gate_b_pass": gate_b_pass,
        "details": {
            "split": split,
            "sample_games": n_use,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "ablation": {
                "baseline_auc_m15": swap_abl["baseline_auc_m15"],
                "swap_auc_m15": swap_abl["swap_auc_m15"],
                "no_rank_auc_m15": swap_abl["no_rank_auc_m15"],
                "rank_only_auc_m15": swap_abl["rank_only_auc_m15"],
                "n_games_swap": swap_abl["n_games_swap"],
            },
            "band_stratified": band_stratified,
            "stratification_counts": stratification_counts,
            "per_band_cap": per_band_cap if band_stratified else None,
            "collapse": collapse,
            "band_scheme": {
                "names": BAND_NAMES,
                "tier_to_band": {str(k): v for k, v in TIER_TO_BAND.items()},
            },
            "classification_thresholds": {
                "under_use": "auc(player_emb) >= 0.70 and auc(h_t) < 0.60",
                "over_leak": (
                    "auc(h_t) >= 0.75 and auc(post_mu) >= 0.70 "
                    "and swap_delta >= 0.05"
                ),
                "else": "mixed_or_unclear",
            },
        },
    }

    print(
        f"[rank-diagnose] classification={classification} gate_b_pass={gate_b_pass}",
        flush=True,
    )
    write_rank_diagnosis_artifact(artifact_path, metrics)
    return metrics
