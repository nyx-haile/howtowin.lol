"""Step 5 / Gate E — causal validation filter for surfaced candidates.

Reads ``artifacts/intervention_candidates.json`` (Step 4 output), aggregates
treated/control rows from a holdout sample, and applies three estimators
(propensity-score matching + DR/AIPW + DML) per (decision_type, anchor_minute)
bucket. Writes ``artifacts/causal_filter_report.json``.

The unit of analysis is *(game, anchor, team)*. Each anchor produces two rows
— one per team — with covariates ``X = [h_t || mu_q(z_t) || macro_t || team]``
and outcome ``Y = team won``. Treatment ``T_dt`` is "≥1 event of type ``dt`` by
this team in the anchor's event window".
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games
from model.intervention import INTERVENTION_DECISION_TYPES
from model.intervention_driver import _band_label
from model.rank_band import game_band
from model.retrieval import MID_GAME_MINUTES
from model.tokens import EVENT_TYPE_TO_ID

# Gate E thresholds. The "binary presence" treatment is structurally
# degenerate for events that happen nearly every minute (ITEM_PURCHASED) or
# almost never (ENGAGE/DISENGAGE in sparse anchor windows). The validator
# does not try to fix that — it transparently rejects pairs with no usable
# overlap. Gate E passes iff *some* candidates clear the bar AND the filter
# isn't completely overlap-bound.
OVERLAP_FLOOR: float = 0.20        # min fraction of rows in propensity common support
MIN_PER_ARM: int = 15              # minimum rows in each treatment arm
PROPENSITY_TRIM: float = 0.05      # propensity clipping for AIPW stability
SIGNIFICANCE_T: float = 1.96       # |t| above this counts as a significant contradiction
GATE_E_MIN_ACCEPTS_TOTAL: int = 1  # ≥1 candidate accepted overall
GATE_E_MAX_OVERLAP_FAIL_RATE: float = 0.95  # stop condition: > this fraction
                                            # rejected for overlap means upstream issue


def _t_stat(effect: float, se: float) -> float:
    if se <= 0 or not np.isfinite(se):
        return 0.0
    return float(effect / se)


def _cluster_meat(scores: np.ndarray, clusters: Optional[np.ndarray]) -> float:
    """Sum of squared cluster-sums of ``scores``.

    With ``clusters=None`` reduces to ``sum(scores ** 2)``, the IID middle
    term. The point of clustering is to absorb within-cluster outcome
    correlation that would otherwise deflate SEs.
    """
    if clusters is None:
        return float((scores ** 2).sum())
    unique = np.unique(clusters)
    total = 0.0
    for c in unique:
        s = float(scores[clusters == c].sum())
        total += s * s
    return float(total)


def _cross_fit_propensity(X: np.ndarray, T: np.ndarray, K: int = 5,
                          random_state: int = 2) -> np.ndarray:
    """Out-of-fold propensity ``P(T=1|X)`` for matching/AIPW overlap.

    In-sample propensities overfit on high-D ``[h_t || mu_q(z_t) || macro]``
    covariates and inflate the overlap diagnostic. Cross-fitting fixes this
    at small extra cost (one logistic per fold).
    """
    n = len(T)
    p = np.zeros(n, dtype=np.float64)
    if n < 2 * K:
        # Fall back to in-sample if data too small to fold.
        lr = LogisticRegression(max_iter=500, solver="liblinear")
        lr.fit(X, T)
        return lr.predict_proba(X)[:, 1]
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        lr = LogisticRegression(max_iter=500, solver="liblinear")
        lr.fit(X[tr], T[tr])
        p[te] = lr.predict_proba(X[te])[:, 1]
    return p


def _matching_estimator(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                        cluster_ids: Optional[np.ndarray] = None) -> dict:
    """Propensity-score 1-NN ATT with cross-fit propensity.

    The propensity is fit out-of-fold so the overlap diagnostic and the NN
    distances aren't optimistic. Matching is with replacement (a control
    can match many treated rows), so the paired-difference SE strictly
    understates the true matching variance — the Abadie-Imbens (2006)
    correction is not applied here. ``cluster_ids`` (one per row) absorbs
    within-game outcome correlation by clustering the paired-difference
    SE on the treated rows' clusters.
    """
    n_t = int((T == 1).sum())
    n_c = int((T == 0).sum())
    if n_t == 0 or n_c == 0:
        return {"effect": 0.0, "se": float("inf"), "overlap": 0.0,
                "n_treated": n_t, "n_control": n_c}

    p = _cross_fit_propensity(X, T)
    overlap = float(((p >= PROPENSITY_TRIM) & (p <= 1 - PROPENSITY_TRIM)).mean())

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    diffs = np.abs(p[treated_idx][:, None] - p[control_idx][None, :])
    nn = np.argmin(diffs, axis=1)
    matched_y = Y[control_idx[nn]]
    paired = Y[treated_idx] - matched_y
    effect = float(paired.mean())

    treated_clusters = (
        cluster_ids[treated_idx] if cluster_ids is not None else None
    )
    centered = paired - effect
    meat = _cluster_meat(centered, treated_clusters)
    se = float(np.sqrt(meat) / n_t) if n_t > 0 else float("inf")
    return {
        "effect": effect,
        "se": se,
        "overlap": overlap,
        "n_treated": n_t,
        "n_control": n_c,
    }


def _fit_outcome_arm(X_tr: np.ndarray, Y_tr: np.ndarray, T_tr: np.ndarray, arm: int) -> Ridge:
    """Fit the outcome model restricted to one treatment arm.

    Falls back to predicting the marginal mean if the arm is empty in this
    fold (rare with cross-fitting + balanced data, but safe).
    """
    mask = T_tr == arm
    m = Ridge(alpha=1.0)
    if mask.sum() < 2:
        m.fit(X_tr, np.full(len(X_tr), float(Y_tr.mean()) if len(Y_tr) else 0.0))
    else:
        m.fit(X_tr[mask], Y_tr[mask])
    return m


def _dr_estimator(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                  cluster_ids: Optional[np.ndarray] = None, K: int = 5,
                  random_state: int = 0) -> dict:
    """AIPW (doubly robust) ATE with K-fold cross-fitting.

    ``cluster_ids`` (one per row) gives a cluster-robust SE for the
    influence-function score ``psi``; without it, the SE is the IID
    ``psi.std/√n``. ``random_state`` controls the KFold shuffle seed and
    must match :func:`_dml_estimator` if the caller wants the two
    estimators to be "two views of the same fold structure".
    """
    n = len(Y)
    if n < 2 * K:
        return {"effect": 0.0, "se": float("inf")}
    psi = np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        prop = LogisticRegression(max_iter=500, solver="liblinear")
        prop.fit(X[tr], T[tr])
        p_te = np.clip(prop.predict_proba(X[te])[:, 1], PROPENSITY_TRIM, 1 - PROPENSITY_TRIM)
        m1 = _fit_outcome_arm(X[tr], Y[tr], T[tr], arm=1)
        m0 = _fit_outcome_arm(X[tr], Y[tr], T[tr], arm=0)
        m1_te = m1.predict(X[te])
        m0_te = m0.predict(X[te])
        psi[te] = (
            m1_te - m0_te
            + T[te] * (Y[te] - m1_te) / p_te
            - (1 - T[te]) * (Y[te] - m0_te) / (1 - p_te)
        )
    effect = float(psi.mean())
    centered = psi - effect
    meat = _cluster_meat(centered, cluster_ids)
    se = float(np.sqrt(meat) / n) if n > 0 else float("inf")
    return {"effect": effect, "se": se}


def _dml_estimator(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                   cluster_ids: Optional[np.ndarray] = None, K: int = 5,
                   random_state: int = 0) -> dict:
    """DML (Robinson partialling-out) ATE with K-fold cross-fitting.

    Cluster-robust sandwich: the meat term ``sum (T_res * resid)²`` is
    replaced with the sum of squared cluster-sums of ``T_res * resid``.
    ``random_state`` controls the KFold shuffle seed and is shared with
    :func:`_dr_estimator` by default so DR and DML see the same fold
    splits, making their effect/SE pair directly comparable.
    """
    n = len(Y)
    if n < 2 * K:
        return {"effect": 0.0, "se": float("inf")}
    Y_res = np.zeros(n)
    T_res = np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        Y_res[te] = Y[te] - m_y.predict(X[te])
        m_t = Ridge(alpha=1.0).fit(X[tr], T[tr].astype(np.float64))
        T_res[te] = T[te] - m_t.predict(X[te])
    denom = float((T_res ** 2).sum())
    if denom < 1e-9:
        return {"effect": 0.0, "se": float("inf")}
    theta = float((T_res * Y_res).sum() / denom)
    resid = Y_res - theta * T_res
    score = T_res * resid
    meat = _cluster_meat(score, cluster_ids)
    var = meat / (denom ** 2)
    return {"effect": theta, "se": float(np.sqrt(max(var, 0.0)))}


def _agreement(estimates: list[dict]) -> tuple[bool, str]:
    """Acceptance rule for the three estimators.

    Two checks must both pass:

    1. **No significant contradiction.** An estimator with opposite sign to
       the majority direction and ``|t| > SIGNIFICANCE_T`` rejects the pair.
    2. **Positive evidence.** At least one estimator must reach
       ``|t| >= SIGNIFICANCE_T``. Without this, "the three estimators
       happen to agree on a direction by chance" — a noisy null — would
       count as a Gate E pass. The plan's "agree on direction OR do not
       contradict" is meant to *not reject* such pairs on disagreement
       grounds; it is not meant to *accept* them as validated.
    """
    signs = [int(np.sign(e["effect"])) for e in estimates]
    nonzero = {s for s in signs if s != 0}

    if len(nonzero) > 1:
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        majority = 1 if pos >= neg else -1
        for est, s in zip(estimates, signs):
            if s == -majority:
                t = abs(_t_stat(est["effect"], est.get("se", 0.0)))
                if t > SIGNIFICANCE_T:
                    return False, f"estimator contradicts (t={t:.2f})"

    max_t = max(
        abs(_t_stat(e["effect"], e.get("se", 0.0))) for e in estimates
    )
    if max_t < SIGNIFICANCE_T:
        return False, f"underpowered (max|t|={max_t:.2f} < {SIGNIFICANCE_T})"

    return True, "agree" if len(nonzero) <= 1 else "no significant contradiction"


def _confidence(estimates: list[dict]) -> float:
    """Mean |effect| divided by mean SE (a stabilised |t|).

    Returns 0 when SEs are degenerate. Caller treats higher as more
    confident; ``> SIGNIFICANCE_T`` is "statistically meaningful".
    """
    eff = np.mean([abs(e["effect"]) for e in estimates])
    ses = [e.get("se", 0.0) for e in estimates if np.isfinite(e.get("se", float("inf")))]
    if not ses:
        return 0.0
    se_mean = float(np.mean(ses))
    if se_mean <= 0:
        return 0.0
    return float(eff / se_mean)


def evaluate_pair(rows: list[dict], dt: str, minute: int) -> dict:
    """Run all three estimators for one (decision_type, anchor_minute) pair.

    Cluster-robust SEs are computed on ``game_idx`` when present in rows.
    The blue/red row pair for a single (game, anchor) has perfectly
    anti-correlated outcomes; without clustering, the IID SE underestimates
    the true variance roughly by ``√2``.
    """
    sel = [r for r in rows if r["minute"] == minute]
    if not sel:
        return {"accept": False, "reason": "no rows at this minute",
                "diagnostics": {"n_treated": 0, "n_control": 0}}

    X = np.stack([r["x"] for r in sel]).astype(np.float64, copy=False)
    T = np.array([r["treatments"][dt] for r in sel], dtype=np.int64)
    Y = np.array([r["outcome"] for r in sel], dtype=np.float64)
    cluster_ids = (
        np.array([r["game_idx"] for r in sel], dtype=np.int64)
        if all("game_idx" in r for r in sel) else None
    )

    n_t = int((T == 1).sum())
    n_c = int((T == 0).sum())
    if n_t < MIN_PER_ARM or n_c < MIN_PER_ARM:
        return {
            "accept": False,
            "reason": f"insufficient overlap (n_t={n_t}, n_c={n_c}, min={MIN_PER_ARM})",
            "diagnostics": {"n_treated": n_t, "n_control": n_c},
        }

    matching = _matching_estimator(X, T, Y, cluster_ids=cluster_ids)
    if matching["overlap"] < OVERLAP_FLOOR:
        return {
            "accept": False,
            "reason": f"propensity overlap {matching['overlap']:.2f} < floor {OVERLAP_FLOOR}",
            "diagnostics": {"matching": matching, "n_treated": n_t, "n_control": n_c},
        }

    dr = _dr_estimator(X, T, Y, cluster_ids=cluster_ids)
    dml = _dml_estimator(X, T, Y, cluster_ids=cluster_ids)
    # Acceptance evidence is computed from DR + DML only. Matching's
    # paired-difference SE understates true variance: with-replacement NN
    # has uncorrected multiplicity (Abadie-Imbens 2006) and the cluster
    # term sees only treated rows, missing cross-game control reuse. The
    # matching point estimate is still emitted for cross-estimator sanity
    # checks, but the gating |t|≥1.96 rule runs against DR / DML so it
    # isn't paid for by an over-tight SE.
    agree, reason = _agreement([dr, dml])
    confidence = _confidence([dr, dml])
    max_t = max(
        abs(_t_stat(e["effect"], e.get("se", 0.0)))
        for e in (dr, dml)
    )
    return {
        "accept": bool(agree),
        "reason": reason,
        "confidence": confidence,
        "max_t_stat": max_t,
        "effects": {
            "matching": float(matching["effect"]),
            "dr": float(dr["effect"]),
            "dml": float(dml["effect"]),
        },
        "ses": {
            "matching": float(matching["se"]),
            "dr": float(dr["se"]),
            "dml": float(dml["se"]),
        },
        "diagnostics": {
            "matching": matching,
            "dr": dr,
            "dml": dml,
            "n_treated": n_t,
            "n_control": n_c,
        },
    }


@torch.no_grad()
def extract_rows(
    *,
    model,
    holdout_match_ids: list[str],
    puuid_index: dict,
    exclude_match_ids: set[str],
    device: str = "cpu",
    minutes: Iterable[int] = MID_GAME_MINUTES,
    max_games: Optional[int] = None,
) -> list[dict]:
    """Encode each game; emit two rows per qualifying anchor (one per team).

    Each row carries ``x`` (covariates), per-decision-type treatment flags,
    binary outcome, and rank band. The covariate vector is
    ``[h_t || post_mu_t || macro_t || team_indicator]``.
    """
    ids = holdout_match_ids[:max_games] if max_games else holdout_match_ids
    minute_set = {int(m) for m in minutes}
    dt_type_ids = {dt: EVENT_TYPE_TO_ID[dt] for dt in INTERVENTION_DECISION_TYPES}

    ds = MatchDataset(
        ids, puuid_index=puuid_index,
        exclude_match_ids=exclude_match_ids, cache_size=1,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)

    rows: list[dict] = []
    for gi, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)

        h_seq = out["h"][0]
        post_mu_seq = out["post_mu"][0]
        macro_seq = batch["anchor_macro_features"][0]
        anchor_pos = batch["anchor_positions"][0].long()
        token_ts = batch["token_timestamps"][0]
        anchor_ts = token_ts.gather(0, anchor_pos)
        minutes_per = (anchor_ts.float() / 60000.0).round().to(torch.int64).cpu().tolist()

        ev_offsets = batch["event_window_offsets"][0]
        ev_counts = batch["event_window_counts"][0]
        ev_positions = batch["event_window_positions"]
        token_types = batch["tokens"][0]
        token_actors = batch["token_actors"][0]

        outcome = float(batch["outcome"][0].item())
        n_valid = int(batch["anchor_mask"][0].sum().item())
        rank_band = _band_label(int(game_band(batch["players"])[0].item()))

        for ai in range(n_valid):
            minute = int(minutes_per[ai])
            if minute not in minute_set:
                continue
            count = int(ev_counts[ai].item())
            if count <= 0:
                continue
            offset = int(ev_offsets[ai].item())
            window_pos = ev_positions[offset:offset + count].long()
            window_types = token_types[window_pos]
            window_actors = token_actors[window_pos]

            blue = (window_actors >= 1) & (window_actors <= 5)
            red = (window_actors >= 6) & (window_actors <= 10)

            treatments_blue = {
                dt: int(((window_types == tid) & blue).any().item())
                for dt, tid in dt_type_ids.items()
            }
            treatments_red = {
                dt: int(((window_types == tid) & red).any().item())
                for dt, tid in dt_type_ids.items()
            }

            x_base = np.concatenate([
                h_seq[ai].cpu().numpy().ravel(),
                post_mu_seq[ai].cpu().numpy().ravel(),
                macro_seq[ai].cpu().numpy().ravel(),
            ]).astype(np.float32)
            x_blue = np.concatenate([x_base, np.array([1.0], dtype=np.float32)])
            x_red = np.concatenate([x_base, np.array([0.0], dtype=np.float32)])

            rows.append({
                "game_idx": gi, "minute": minute, "team": "blue",
                "x": x_blue, "treatments": treatments_blue,
                "outcome": outcome, "rank_band": rank_band,
            })
            rows.append({
                "game_idx": gi, "minute": minute, "team": "red",
                "x": x_red, "treatments": treatments_red,
                "outcome": 1.0 - outcome, "rank_band": rank_band,
            })
    return rows


def _candidate_pairs(candidates: list[dict]) -> list[tuple[str, int]]:
    """Unique ``(decision_type, anchor_minute)`` pairs from Step 4 output."""
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for c in candidates:
        key = (c["decision_type"], int(c["anchor_minute"]))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _build_caveat(rank_bands_seen: list[str], corpus_tag: str) -> str:
    """Construct the ``_caveat`` string from runtime evidence.

    Avoids hard-coded "51k corpus / master_plus" claims that go stale on a
    rebalanced rerun. The caveat reports the corpus tag (if provided) and
    classifies the rank coverage of the candidate set:

    - 0 bands: empty result; no rank evidence at all.
    - 1 band: cross-band rank-confounding check is vacuous.
    - ≥2 bands: cross-band check is meaningful.
    """
    tag_clause = f"corpus={corpus_tag}; " if corpus_tag else ""
    if not rank_bands_seen:
        return (
            f"{tag_clause}no rank bands appeared in the candidate set, so "
            "rank-confounding evidence is unavailable. Treat any per-pair "
            "estimates as preliminary."
        )
    bands_str = ", ".join(rank_bands_seen)
    if len(rank_bands_seen) == 1:
        return (
            f"{tag_clause}candidate set is concentrated in a single rank "
            f"band ({bands_str}); the cross-band rank-confounding check is "
            "therefore vacuous and Gate E within-band evidence stands only "
            "for that band. Cross-band validation requires a more rank-"
            "balanced corpus."
        )
    return (
        f"{tag_clause}candidate set spans rank bands {bands_str}; cross-"
        "band rank-confounding check is non-vacuous. Per-band accept rates "
        "should still be inspected for skew before promoting candidates to "
        "the lesson surface."
    )


def run_causal_filter(
    *,
    model,
    holdout_match_ids: list[str],
    puuid_index: dict,
    exclude_match_ids: set[str],
    candidates: list[dict],
    device: str = "cpu",
    max_games: Optional[int] = None,
    corpus_tag: str = "",
) -> dict:
    """End-to-end: extract rows, evaluate every candidate pair, build report.

    ``corpus_tag`` is an optional short label for the corpus used in this
    run (e.g. ``"51k-local"`` or ``"cloud-rebalanced-v1"``). It surfaces in
    the report's ``_caveat`` field so downstream readers can tell which
    corpus shape produced the rank-band pattern.
    """
    rows = extract_rows(
        model=model,
        holdout_match_ids=holdout_match_ids,
        puuid_index=puuid_index,
        exclude_match_ids=exclude_match_ids,
        device=device,
        max_games=max_games,
    )

    pairs = _candidate_pairs(candidates)
    accepted: list[dict] = []
    rejected: list[dict] = []
    accepts_per_type = {dt: 0 for dt in INTERVENTION_DECISION_TYPES}
    overlap_values: list[float] = []
    n_overlap_fail = 0
    n_pairs = len(pairs)
    log_every = max(1, min(20, n_pairs // 20)) if n_pairs else 1
    print(
        f"[causal-filter] evaluating {n_pairs} candidate pairs "
        f"over {len(rows)} rows",
        flush=True,
    )

    for pi, (dt, minute) in enumerate(pairs, start=1):
        result = evaluate_pair(rows, dt, minute)
        diag = result.get("diagnostics", {})
        m = diag.get("matching") if isinstance(diag, dict) else None
        if isinstance(m, dict) and "overlap" in m:
            overlap_values.append(float(m["overlap"]))

        reason = result.get("reason", "")
        if "overlap" in reason or "insufficient" in reason:
            n_overlap_fail += 1

        if result["accept"]:
            diag_n_t = diag.get("n_treated") if isinstance(diag, dict) else None
            diag_n_c = diag.get("n_control") if isinstance(diag, dict) else None
            accepted.append({
                "decision_type": dt,
                "anchor_minute": minute,
                "confidence": float(result.get("confidence", 0.0)),
                "max_t_stat": float(result.get("max_t_stat", 0.0)),
                "effects": result.get("effects", {}),
                "ses": result.get("ses", {}),
                "n_treated": int(diag_n_t) if diag_n_t is not None else None,
                "n_control": int(diag_n_c) if diag_n_c is not None else None,
            })
            accepts_per_type[dt] += 1
        else:
            rejected.append({
                "decision_type": dt,
                "anchor_minute": minute,
                "reason": result["reason"],
            })

        if pi % log_every == 0 or pi == n_pairs:
            print(
                f"[causal-filter] {pi}/{n_pairs} pairs evaluated  "
                f"accepted={len(accepted)}  rejected={len(rejected)}",
                flush=True,
            )

    overlap_median = float(np.median(overlap_values)) if overlap_values else 0.0
    overlap_fail_rate = n_overlap_fail / len(pairs) if pairs else 1.0
    stop_condition = overlap_fail_rate > GATE_E_MAX_OVERLAP_FAIL_RATE
    rank_bands_seen = sorted({r["rank_band"] for r in rows}) if rows else []

    # Within-band: at least one accept and the filter isn't structurally
    # overlap-bound. Cross-band: also requires evidence from ≥2 rank bands,
    # otherwise the rank-confounding check is vacuous (B2 — verifier finding).
    gate_e_pass_within_band = bool(
        len(accepted) >= GATE_E_MIN_ACCEPTS_TOTAL
        and not stop_condition
    )
    gate_e_pass = bool(gate_e_pass_within_band and len(rank_bands_seen) >= 2)

    return {
        "_caveat": _build_caveat(rank_bands_seen, corpus_tag),
        "accepted": accepted,
        "rejected": rejected,
        "gate_e_pass": gate_e_pass,
        "gate_e_pass_within_band": gate_e_pass_within_band,
        "summary": {
            "n_pairs_evaluated": len(pairs),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            "accepts_per_type": accepts_per_type,
            "overlap_median": overlap_median,
            "overlap_fail_rate": overlap_fail_rate,
            "stop_condition_triggered": stop_condition,
            "n_rows": len(rows),
            "rank_bands_seen": rank_bands_seen,
            "overlap_floor": OVERLAP_FLOOR,
            "min_per_arm": MIN_PER_ARM,
            "significance_t": SIGNIFICANCE_T,
        },
    }


def write_causal_filter_report(path: str, payload: dict) -> None:
    """Atomic JSON write (mirrors intervention_driver helper)."""
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dirn or ".", prefix=".causal_filter_report.",
        suffix=".tmp", delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
