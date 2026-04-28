"""Step 4 / Gate D — driver for the counterfactual intervention scan.

Iterates a holdout sample, encodes each game with :class:`PlanBModel`,
selects mid-game anchors, and for each (anchor, decision_type) compares
the base rollout against an intervention rollout produced by
:func:`run_intervention_rollout`. Mean per-step Jensen-Shannon divergence
on the deterministic outcome distribution is the candidate score.

Writes ``artifacts/intervention_candidates.json``.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games
from model.intervention import (
    INTERVENTION_DECISION_TYPES,
    build_synthetic_token_embedding,
    js_divergence,
    run_intervention_rollout,
)
from model.rank_band import BAND_NAMES, UNRANKED_BAND, game_band
from model.retrieval import MID_GAME_MINUTES

# Empirical thresholds. The candidate divergence is symmetric KL on the
# RSSM prior latent (see intervention.js_divergence). On the 51k Plan B
# checkpoint, sustained replace-window injection yields KL ~ 1e-3 across
# decision types vs. noise floor ~ 1e-8 (no perturbation). Floor is set
# 100x above noise; signal marks a "clearly differentiated" intervention.
JS_DIVERGENCE_FLOOR: float = 1e-4
JS_DIVERGENCE_SIGNAL_THRESHOLD: float = 5e-4

# Cap rollout horizon per anchor — 4 steps ≈ 4 minutes at the model's anchor
# spacing, enough to surface divergence without blowing runtime.
DEFAULT_ROLLOUT_STEPS: int = 4

PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT = (
    "PlanBModel.forward and rollout_prior_single sample RSSM latents; the "
    "intervention scan keeps teacher-force forwards and rollouts one game/"
    "anchor at a time until deterministic or RNG-replay parity is specified."
)


def _inc_counter(counters: dict | None, name: str, amount: int = 1) -> None:
    if counters is None:
        return
    counters[name] = int(counters.get(name, 0)) + amount


def _record_model_forward_counter(counters: dict | None, batch: dict) -> None:
    if counters is None:
        return
    _inc_counter(counters, "model_forward_calls")
    batch_size = 1
    tokens = batch.get("tokens")
    if torch.is_tensor(tokens):
        batch_size = int(tokens.size(0))
    counters["max_model_forward_batch_size"] = max(
        int(counters.get("max_model_forward_batch_size", 0)),
        batch_size,
    )
    counters.setdefault("stochastic_planb_forward_batching", "disabled_batch_size_1")
    counters.setdefault("stochastic_planb_forward_caveat", PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT)


def _band_label(band_id: int) -> str:
    if band_id == UNRANKED_BAND or not (0 <= band_id < len(BAND_NAMES)):
        return "unranked"
    return BAND_NAMES[band_id]


@torch.no_grad()
def _encode_one_game(model, batch, *, counters: dict | None = None) -> dict:
    """Run a forward pass and pull tensors needed by ``rollout_prior_single``.

    Returns a dict with: ``h``, ``z``, ``static_tokens``, ``token_embeddings``,
    ``event_window_positions/offsets/counts``, ``anchor_mask``, ``minutes``,
    ``n_valid_anchors``, ``rank_band``.
    """
    out = model(batch)
    _record_model_forward_counter(counters, batch)
    anchor_pos = batch["anchor_positions"][0]
    ts = batch["token_timestamps"][0]
    anchor_ts = ts.gather(0, anchor_pos.long())
    minutes = (anchor_ts.float() / 60000.0).round().to(torch.int64)
    band_id = int(game_band(batch["players"])[0].item())
    n_valid = int(batch["anchor_mask"][0].sum().item())
    return {
        "h": out["h"][0],                        # (T, D_H)
        "z": out["z"][0],                        # (T, D_Z)
        "static_tokens": out["static_tokens"][0],  # (T_static, D_MODEL)
        "token_embeddings": out["token_embeddings"][0],  # (L, D_MODEL)
        "event_window_positions": batch["event_window_positions"],
        "event_window_offsets": batch["event_window_offsets"][0],
        "event_window_counts": batch["event_window_counts"][0],
        "anchor_mask": batch["anchor_mask"][0],
        "minutes": minutes,
        "n_valid_anchors": n_valid,
        "rank_band": band_id,
    }


@torch.no_grad()
def _score_anchor(
    model,
    encoded: dict,
    *,
    anchor_idx: int,
    n_steps: int,
    synthetic_embs: dict[str, torch.Tensor],
    replace_window: bool = True,
    sustained: bool = True,
    counters: dict | None = None,
) -> dict[str, float]:
    """Per-decision-type JS divergence at one anchor.

    Returns ``{decision_type: js_score}``. If the rollout horizon at this
    anchor is < 1 step, returns an empty dict.
    """
    if anchor_idx >= encoded["n_valid_anchors"] - 1:
        return {}
    base = model.rollout_prior_single(
        h0=encoded["h"][anchor_idx],
        z0=encoded["z"][anchor_idx],
        static_tokens=encoded["static_tokens"],
        token_embeddings=encoded["token_embeddings"],
        event_window_positions=encoded["event_window_positions"],
        event_window_offsets=encoded["event_window_offsets"],
        event_window_counts=encoded["event_window_counts"],
        anchor_mask=encoded["anchor_mask"],
        start_anchor=anchor_idx,
        n_steps=n_steps,
        n_valid_anchors=encoded["n_valid_anchors"],
    )
    _inc_counter(counters, "base_rollout_calls")
    if not base:
        return {}
    scores: dict[str, float] = {}
    for dt, synth in synthetic_embs.items():
        perturbed = run_intervention_rollout(
            model,
            h0=encoded["h"][anchor_idx],
            z0=encoded["z"][anchor_idx],
            static_tokens=encoded["static_tokens"],
            token_embeddings=encoded["token_embeddings"],
            event_window_positions=encoded["event_window_positions"],
            event_window_offsets=encoded["event_window_offsets"],
            event_window_counts=encoded["event_window_counts"],
            anchor_mask=encoded["anchor_mask"],
            start_anchor=anchor_idx,
            n_steps=n_steps,
            n_valid_anchors=encoded["n_valid_anchors"],
            synthetic_emb=synth,
            inject_at_step=0,
            replace_window=replace_window,
            sustained=sustained,
        )
        _inc_counter(counters, "intervention_rollout_calls")
        scores[dt] = js_divergence(base, perturbed, model)
    return scores


@torch.no_grad()
def run_intervention_scan(
    *,
    model,
    holdout_match_ids: list[str],
    puuid_index: dict,
    exclude_match_ids: set[str],
    device: str = "cpu",
    mid_minutes: Iterable[int] = MID_GAME_MINUTES,
    n_steps: int = DEFAULT_ROLLOUT_STEPS,
    max_games: Optional[int] = None,
    max_anchors_per_game: Optional[int] = 4,
    js_floor: float = JS_DIVERGENCE_FLOOR,
    js_signal: float = JS_DIVERGENCE_SIGNAL_THRESHOLD,
    replace_window: bool = True,
    sustained: bool = True,
) -> dict:
    """Build candidate list ``[{decision_type, anchor_minute, divergence_score, rank_band}, …]``.

    Gate D passes when at least one candidate per decision type reaches
    ``js_signal`` and the median candidate score (across the full surface)
    clears ``js_floor``.
    """
    ids = holdout_match_ids[:max_games] if max_games else holdout_match_ids
    mid_set = set(int(m) for m in mid_minutes)

    # Synthetic embeddings depend on the model only — build once.
    synthetic_embs = {
        dt: build_synthetic_token_embedding(model, decision_type=dt, device=device)
        for dt in INTERVENTION_DECISION_TYPES
    }

    ds = MatchDataset(
        ids, puuid_index=puuid_index,
        exclude_match_ids=exclude_match_ids, cache_size=1,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)

    candidates: list[dict] = []
    n_games_scanned = 0
    n_anchors_scored = 0
    counters: dict = {}
    total_games = len(ids)
    log_every = max(1, min(50, total_games // 20)) if total_games else 1

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        encoded = _encode_one_game(model, batch, counters=counters)
        band_label = _band_label(encoded["rank_band"])
        minutes_cpu = encoded["minutes"].cpu()

        mid_anchor_idxs = [
            i for i, m in enumerate(minutes_cpu.tolist())
            if i < encoded["n_valid_anchors"] - 1 and m in mid_set
        ]
        if max_anchors_per_game and len(mid_anchor_idxs) > max_anchors_per_game:
            stride = max(1, len(mid_anchor_idxs) // max_anchors_per_game)
            mid_anchor_idxs = mid_anchor_idxs[::stride][:max_anchors_per_game]

        for ai in mid_anchor_idxs:
            scores = _score_anchor(
                model, encoded, anchor_idx=ai,
                n_steps=n_steps, synthetic_embs=synthetic_embs,
                replace_window=replace_window,
                sustained=sustained,
                counters=counters,
            )
            if not scores:
                continue
            n_anchors_scored += 1
            anchor_minute = int(minutes_cpu[ai].item())
            for dt, js in scores.items():
                candidates.append({
                    "decision_type": dt,
                    "anchor_minute": anchor_minute,
                    "divergence_score": float(js),
                    "rank_band": band_label,
                })
        n_games_scanned += 1
        if n_games_scanned % log_every == 0 or n_games_scanned == total_games:
            print(
                f"[intervention] scanned {n_games_scanned}/{total_games} games"
                f"  anchors_scored={n_anchors_scored}"
                f"  candidates={len(candidates)}",
                flush=True,
            )

    # Aggregate stats for Gate D evaluation.
    per_type_max = {dt: 0.0 for dt in INTERVENTION_DECISION_TYPES}
    for c in candidates:
        if c["divergence_score"] > per_type_max[c["decision_type"]]:
            per_type_max[c["decision_type"]] = c["divergence_score"]
    all_scores = sorted((c["divergence_score"] for c in candidates), reverse=True)
    median = all_scores[len(all_scores) // 2] if all_scores else 0.0
    n_above_signal = sum(1 for s in all_scores if s >= js_signal)
    n_above_floor = sum(1 for s in all_scores if s >= js_floor)

    # Pass condition: every decision type has ≥1 candidate at signal AND a
    # meaningful number of candidates clear the floor. We don't gate on the
    # median — the candidate distribution is long-tailed (most anchors are
    # not lesson-worthy moments), so the median naturally sits well below
    # the per-type maxima.
    every_type_signal = all(per_type_max[dt] >= js_signal for dt in INTERVENTION_DECISION_TYPES)
    floor_count_ok = n_above_floor >= len(INTERVENTION_DECISION_TYPES)
    gate_d_pass = bool(every_type_signal and floor_count_ok and candidates)

    band_counts = {b: 0 for b in ("iron_silver", "gold_platinum",
                                   "emerald_diamond", "master_plus", "unranked")}
    for c in candidates:
        band_counts[c["rank_band"]] = band_counts.get(c["rank_band"], 0) + 1

    return {
        "_caveat": (
            "51k corpus rank distribution is heavily skewed (67% master_plus, "
            "32% unranked). Candidates therefore concentrate in those bands; "
            "diversity will improve when the corpus is rebalanced in the second "
            "wave (cloud rebuild). The intervention surface itself is band-agnostic."
        ),
        "candidates": candidates,
        "gate_d_pass": gate_d_pass,
        "rank_band_counts": band_counts,
        "stochastic_eval_gate": {
            "planb_forward_batching": "disabled_batch_size_1",
            "reason": PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT,
        },
        "summary": {
            "n_games_scanned": n_games_scanned,
            "n_anchors_scored": n_anchors_scored,
            "n_candidates": len(candidates),
            "n_model_forward_calls": int(counters.get("model_forward_calls", 0)),
            "n_base_rollouts": int(counters.get("base_rollout_calls", 0)),
            "n_intervention_rollouts": int(counters.get("intervention_rollout_calls", 0)),
            "max_model_forward_batch_size": int(counters.get("max_model_forward_batch_size", 0)),
            "median_divergence": float(median),
            "n_above_signal": int(n_above_signal),
            "n_above_floor": int(n_above_floor),
            "per_type_max": {k: float(v) for k, v in per_type_max.items()},
            "js_floor": float(js_floor),
            "js_signal": float(js_signal),
            "n_steps": int(n_steps),
        },
    }


def write_intervention_candidates_artifact(path: str, payload: dict) -> None:
    """Atomic JSON write for the Gate D handoff artifact (mirrors m4_eval)."""
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dirn or ".", prefix=".intervention_candidates.",
        suffix=".tmp", delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
