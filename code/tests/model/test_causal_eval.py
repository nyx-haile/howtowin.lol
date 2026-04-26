"""Step 5 / Gate E — tests for the causal validation filter.

Exercises the three estimators on synthetic data with known ground-truth
ATEs, plus the agreement/confidence helpers and the artifact round-trip.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from model.causal_eval import (
    GATE_E_MAX_OVERLAP_FAIL_RATE,
    GATE_E_MIN_ACCEPTS_TOTAL,
    MIN_PER_ARM,
    OVERLAP_FLOOR,
    SIGNIFICANCE_T,
    _agreement,
    _candidate_pairs,
    _confidence,
    _dml_estimator,
    _dr_estimator,
    _matching_estimator,
    evaluate_pair,
    write_causal_filter_report,
)
from model.intervention import INTERVENTION_DECISION_TYPES


def _synthetic_data(n: int = 600, true_ate: float = 0.3, noise: float = 0.2,
                    confound: bool = True, seed: int = 0):
    """Generate (X, T, Y) with a known ATE.

    With ``confound=True`` the propensity depends on X[:, 0] so naïve
    difference-of-means is biased; the doubly-robust / DML estimators must
    correct for it. Outcome is linear in X plus ``true_ate * T``.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float64)
    if confound:
        logit = 0.7 * X[:, 0] - 0.3 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-logit))
        T = (rng.uniform(size=n) < p).astype(np.int64)
    else:
        T = (rng.uniform(size=n) < 0.5).astype(np.int64)
    Y = 0.5 * X[:, 0] + 0.2 * X[:, 2] + true_ate * T + noise * rng.standard_normal(n)
    return X, T, Y


# --- 1. matching ATT recovers true effect when no confounding -----------------

def test_matching_estimator_recovers_true_effect_no_confound():
    X, T, Y = _synthetic_data(n=800, true_ate=0.4, confound=False, seed=1)
    m = _matching_estimator(X, T, Y)
    assert abs(m["effect"] - 0.4) < 0.15, f"matching ATT off: {m['effect']:.3f} vs 0.4"
    assert m["overlap"] > 0.8, f"overlap {m['overlap']:.2f} should be high under uniform propensity"
    assert m["n_treated"] > 0 and m["n_control"] > 0


# --- 2. matching has propensity-bias under confounding ------------------------

def test_matching_estimator_returns_overlap_diagnostic():
    X, T, Y = _synthetic_data(n=600, true_ate=0.3, confound=True, seed=2)
    m = _matching_estimator(X, T, Y)
    # Overlap can be lower under confounding, but still > 0.
    assert 0.0 < m["overlap"] <= 1.0
    assert "n_treated" in m and "n_control" in m


# --- 3. DR / AIPW recovers ATE under confounding ------------------------------

def test_dr_estimator_recovers_true_ate_under_confounding():
    X, T, Y = _synthetic_data(n=800, true_ate=0.5, confound=True, seed=3)
    dr = _dr_estimator(X, T, Y, K=5)
    assert abs(dr["effect"] - 0.5) < 0.15, f"DR off: {dr['effect']:.3f} vs 0.5"
    assert dr["se"] > 0 and np.isfinite(dr["se"])


# --- 4. DML recovers ATE under confounding ------------------------------------

def test_dml_estimator_recovers_true_ate_under_confounding():
    X, T, Y = _synthetic_data(n=800, true_ate=0.5, confound=True, seed=4)
    dml = _dml_estimator(X, T, Y, K=5)
    assert abs(dml["effect"] - 0.5) < 0.15, f"DML off: {dml['effect']:.3f} vs 0.5"
    assert dml["se"] > 0 and np.isfinite(dml["se"])


# --- 5. zero true effect → estimators produce small effect --------------------

def test_estimators_near_zero_when_no_effect():
    X, T, Y = _synthetic_data(n=800, true_ate=0.0, confound=True, seed=5)
    m = _matching_estimator(X, T, Y)
    dr = _dr_estimator(X, T, Y)
    dml = _dml_estimator(X, T, Y)
    for est, name in [(m, "matching"), (dr, "DR"), (dml, "DML")]:
        assert abs(est["effect"]) < 0.2, f"{name} effect too large for zero-ATE data: {est['effect']:.3f}"


# --- 6. _agreement: same sign → agree -----------------------------------------

def test_agreement_when_all_same_sign():
    estimates = [
        {"effect": 0.3, "se": 0.1},
        {"effect": 0.25, "se": 0.05},
        {"effect": 0.4, "se": 0.08},
    ]
    agree, reason = _agreement(estimates)
    assert agree
    assert reason == "agree"


# --- 7. _agreement: opposite sign but not significant → "no contradiction" ----

def test_agreement_opposite_sign_not_significant():
    estimates = [
        {"effect": 0.3, "se": 0.1},
        {"effect": 0.25, "se": 0.05},
        {"effect": -0.05, "se": 0.5},   # opposite sign, but |t|=0.1 < 1.96
    ]
    agree, reason = _agreement(estimates)
    assert agree
    assert "contradiction" in reason or reason == "agree"


# --- 8. _agreement: significant contradiction → reject ------------------------

def test_agreement_significant_contradiction_rejects():
    estimates = [
        {"effect": 0.3, "se": 0.05},
        {"effect": 0.25, "se": 0.05},
        {"effect": -0.5, "se": 0.05},   # |t|=10 contradiction
    ]
    agree, reason = _agreement(estimates)
    assert not agree
    assert "contradicts" in reason


# --- 9. confidence: higher when effects are large relative to SE --------------

def test_confidence_scales_with_t_stat():
    high = [
        {"effect": 0.5, "se": 0.05},
        {"effect": 0.5, "se": 0.05},
        {"effect": 0.5, "se": 0.05},
    ]
    low = [
        {"effect": 0.05, "se": 0.5},
        {"effect": 0.05, "se": 0.5},
        {"effect": 0.05, "se": 0.5},
    ]
    assert _confidence(high) > SIGNIFICANCE_T
    assert _confidence(low) < SIGNIFICANCE_T


# --- 10. evaluate_pair: insufficient overlap → reject -------------------------

def test_evaluate_pair_rejects_insufficient_overlap():
    rng = np.random.default_rng(0)
    rows = []
    # Only a handful of treated rows — below MIN_PER_ARM.
    n_total = MIN_PER_ARM + 100
    for i in range(n_total):
        rows.append({
            "minute": 12,
            "x": rng.standard_normal(8).astype(np.float32),
            "treatments": {dt: 0 for dt in INTERVENTION_DECISION_TYPES},
            "outcome": 1.0 if i % 2 == 0 else 0.0,
        })
    n_treated = max(1, MIN_PER_ARM - 5)
    for i in range(n_treated):
        rows[i]["treatments"]["ITEM_PURCHASED"] = 1

    result = evaluate_pair(rows, "ITEM_PURCHASED", 12)
    assert not result["accept"]
    assert "insufficient overlap" in result["reason"]


# --- 11. evaluate_pair: balanced data → accepts and assigns confidence --------

def test_evaluate_pair_accepts_balanced_data():
    rng = np.random.default_rng(7)
    n = 400
    rows = []
    for i in range(n):
        x = rng.standard_normal(4).astype(np.float32)
        # Balanced treatment, true effect on outcome.
        T_dt = int(rng.uniform() < 0.5)
        Y = float(0.5 + 0.3 * T_dt + 0.2 * rng.standard_normal())
        treatments = {dt: 0 for dt in INTERVENTION_DECISION_TYPES}
        treatments["ITEM_PURCHASED"] = T_dt
        rows.append({"minute": 14, "x": x, "treatments": treatments, "outcome": Y})

    result = evaluate_pair(rows, "ITEM_PURCHASED", 14)
    assert result["accept"], f"expected accept on balanced data, got {result['reason']}"
    assert result["confidence"] > 0.0


# --- 12. _candidate_pairs deduplicates --------------------------------------

def test_candidate_pairs_deduplicates():
    cands = [
        {"decision_type": "ITEM_PURCHASED", "anchor_minute": 10},
        {"decision_type": "ITEM_PURCHASED", "anchor_minute": 10},
        {"decision_type": "ENGAGE", "anchor_minute": 12},
        {"decision_type": "ITEM_PURCHASED", "anchor_minute": 14},
    ]
    pairs = _candidate_pairs(cands)
    assert pairs == [
        ("ITEM_PURCHASED", 10),
        ("ENGAGE", 12),
        ("ITEM_PURCHASED", 14),
    ]


# --- 13. write_causal_filter_report round-trip --------------------------------

def test_write_causal_filter_report_round_trip():
    payload = {
        "_caveat": "test caveat",
        "accepted": [
            {"decision_type": "ITEM_PURCHASED", "anchor_minute": 12, "confidence": 2.5},
        ],
        "rejected": [
            {"decision_type": "ENGAGE", "anchor_minute": 18, "reason": "low overlap"},
        ],
        "gate_e_pass": True,
        "summary": {"n_pairs_evaluated": 2},
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "causal_filter_report.json")
        write_causal_filter_report(path, payload)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == payload


# --- 14. constants are sane ---------------------------------------------------

def test_constants_are_sane():
    assert 0.0 < OVERLAP_FLOOR < 1.0
    assert MIN_PER_ARM >= 10
    assert SIGNIFICANCE_T > 1.5  # ~1.96 → 95% CI
    assert GATE_E_MIN_ACCEPTS_TOTAL >= 1
    assert 0.0 < GATE_E_MAX_OVERLAP_FAIL_RATE <= 1.0
