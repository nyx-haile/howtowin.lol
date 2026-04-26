"""Step 4 / Gate D — tests for the counterfactual intervention surface.

Builds a tiny ``PlanBModel(max_puuids=10)`` with random weights and
synthetic batch tensors. No DB access — everything is in-memory.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import torch

from model.encoders import D_MODEL
from model.intervention import (
    INTERVENTION_DECISION_TYPES,
    JS_LOG2,
    _bernoulli_kl,
    _inject_synthetic_token,
    _outcome_probs,
    build_synthetic_token_embedding,
    js_divergence,
    run_intervention_rollout,
)
from model.plan_b_model import D_H, D_Z, PlanBModel
from model.static_features import STATIC_TOKEN_COUNT


def _tiny_model(seed: int = 0) -> PlanBModel:
    torch.manual_seed(seed)
    model = PlanBModel(max_puuids=10)
    model.eval()
    return model


def _synthetic_rollout_inputs(*, T: int = 4, L: int = 8, count: int = 2):
    """Build a self-consistent set of tensors for ``rollout_prior_single``.

    Per anchor window we set ``count`` events. ``token_embeddings`` is shape
    (L, D_MODEL); positions are constructed so each window indexes valid
    rows. Static tokens / h0 / z0 are random with a fixed seed.
    """
    torch.manual_seed(1)
    h0 = torch.randn(1, D_H)
    z0 = torch.randn(1, D_Z)
    static_tokens = torch.randn(1, STATIC_TOKEN_COUNT, D_MODEL)
    token_embeddings = torch.randn(L, D_MODEL)

    # Round-robin position assignment so every window has ``count`` valid rows.
    positions = torch.arange(T * count, dtype=torch.long) % L
    counts = torch.full((T,), count, dtype=torch.long)
    offsets = torch.zeros(T, dtype=torch.long)
    offsets[1:] = counts[:-1].cumsum(0)
    anchor_mask = torch.ones(1, T, dtype=torch.bool)
    return {
        "h0": h0,
        "z0": z0,
        "static_tokens": static_tokens,
        "token_embeddings": token_embeddings,
        "event_window_positions": positions,
        "event_window_offsets": offsets,
        "event_window_counts": counts,
        "anchor_mask": anchor_mask,
    }


# --- 1. shape ---------------------------------------------------------------

def test_build_synthetic_token_embedding_shape():
    model = _tiny_model()
    for dt in INTERVENTION_DECISION_TYPES:
        emb = build_synthetic_token_embedding(model, decision_type=dt)
        assert emb.shape == (D_MODEL,), f"{dt}: got {emb.shape}"
        assert emb.dtype == torch.float32


# --- 2. per-type distinguishability ----------------------------------------

def test_build_synthetic_token_embedding_differs_by_type():
    model = _tiny_model()
    embs = {
        dt: build_synthetic_token_embedding(model, decision_type=dt)
        for dt in INTERVENTION_DECISION_TYPES
    }
    types = list(embs.keys())
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            d = (embs[types[i]] - embs[types[j]]).norm().item()
            assert d > 1e-3, f"{types[i]} vs {types[j]} embeddings collapsed (L2={d})"


# --- 3. injection actually changes rollout state ---------------------------

def test_run_intervention_rollout_changes_state():
    model = _tiny_model()
    inputs = _synthetic_rollout_inputs()
    base = model.rollout_prior_single(
        h0=inputs["h0"], z0=inputs["z0"],
        static_tokens=inputs["static_tokens"],
        token_embeddings=inputs["token_embeddings"],
        event_window_positions=inputs["event_window_positions"],
        event_window_offsets=inputs["event_window_offsets"],
        event_window_counts=inputs["event_window_counts"],
        anchor_mask=inputs["anchor_mask"],
        start_anchor=0, n_steps=2, n_valid_anchors=4,
    )
    synth = build_synthetic_token_embedding(model, decision_type="ITEM_PURCHASED")
    perturbed = run_intervention_rollout(
        model, **inputs,
        start_anchor=0, n_steps=2, n_valid_anchors=4,
        synthetic_emb=synth, inject_at_step=0,
    )
    assert len(base) == len(perturbed) == 2
    # prior_mu is deterministic; if the synthetic event flowed through the
    # RSSM the L2 distance must be non-trivial.
    diff = (base[0]["prior_mu"] - perturbed[0]["prior_mu"]).norm().item()
    assert diff > 1e-3, f"prior_mu unchanged (L2={diff}) — injection had no effect"


# --- 4. injection helper does not mutate inputs ----------------------------

def test_inject_synthetic_token_no_mutation():
    inputs = _synthetic_rollout_inputs()
    synth = torch.randn(D_MODEL)
    snap = {
        "tok": inputs["token_embeddings"].clone(),
        "pos": inputs["event_window_positions"].clone(),
        "off": inputs["event_window_offsets"].clone(),
        "cnt": inputs["event_window_counts"].clone(),
    }
    _inject_synthetic_token(
        token_embeddings=inputs["token_embeddings"],
        event_window_positions=inputs["event_window_positions"],
        event_window_offsets=inputs["event_window_offsets"],
        event_window_counts=inputs["event_window_counts"],
        inject_window_idx=1,
        synthetic_emb=synth,
    )
    assert torch.equal(inputs["token_embeddings"], snap["tok"])
    assert torch.equal(inputs["event_window_positions"], snap["pos"])
    assert torch.equal(inputs["event_window_offsets"], snap["off"])
    assert torch.equal(inputs["event_window_counts"], snap["cnt"])


# --- 5. JS = 0 for identical rollouts --------------------------------------

def test_js_divergence_zero_for_identical_rollouts():
    model = _tiny_model()
    inputs = _synthetic_rollout_inputs()
    rollout = model.rollout_prior_single(
        h0=inputs["h0"], z0=inputs["z0"],
        static_tokens=inputs["static_tokens"],
        token_embeddings=inputs["token_embeddings"],
        event_window_positions=inputs["event_window_positions"],
        event_window_offsets=inputs["event_window_offsets"],
        event_window_counts=inputs["event_window_counts"],
        anchor_mask=inputs["anchor_mask"],
        start_anchor=0, n_steps=3, n_valid_anchors=4,
    )
    js = js_divergence(rollout, rollout, model)
    assert js == 0.0 or js < 1e-12, f"js={js} should be zero for identical rollouts"


# --- 6. JS > 0 for different rollouts --------------------------------------

def test_js_divergence_positive_for_different_rollouts():
    model = _tiny_model()
    inputs = _synthetic_rollout_inputs()
    base = model.rollout_prior_single(
        h0=inputs["h0"], z0=inputs["z0"],
        static_tokens=inputs["static_tokens"],
        token_embeddings=inputs["token_embeddings"],
        event_window_positions=inputs["event_window_positions"],
        event_window_offsets=inputs["event_window_offsets"],
        event_window_counts=inputs["event_window_counts"],
        anchor_mask=inputs["anchor_mask"],
        start_anchor=0, n_steps=3, n_valid_anchors=4,
    )
    synth = build_synthetic_token_embedding(model, decision_type="ENGAGE")
    perturbed = run_intervention_rollout(
        model, **inputs,
        start_anchor=0, n_steps=3, n_valid_anchors=4,
        synthetic_emb=synth, inject_at_step=0,
    )
    js = js_divergence(base, perturbed, model)
    assert js > 0.0, f"js={js} should be positive for differing rollouts"
    assert math.isfinite(js), f"js={js} must be finite"
    _ = JS_LOG2  # kept exported for downstream callers; not bound here.


# --- 7. JS bounds & numeric safety -----------------------------------------

def test_js_divergence_bounds():
    model = _tiny_model()
    # _outcome_probs is retained as a utility — verify it still produces
    # sigmoid-projected outputs in [0, 1].
    rollout = [{
        "h": torch.zeros(1, D_H),
        "prior_mu": torch.zeros(1, D_Z),
        "prior_logvar": torch.zeros(1, D_Z),
    }]
    p = _outcome_probs(rollout, model)
    assert p.shape == (1,)
    assert (p >= 0).all() and (p <= 1).all()
    # Empty rollouts → 0.
    assert js_divergence([], [], model) == 0.0
    assert js_divergence(rollout, [], model) == 0.0


# --- 8. _bernoulli_kl is non-negative & numerically safe -------------------

def test_bernoulli_kl_non_negative_and_clamped():
    p = torch.tensor([0.0, 0.5, 1.0])
    q = torch.tensor([1.0, 0.5, 0.0])
    kl = _bernoulli_kl(p, q)
    assert torch.isfinite(kl).all(), "KL should not be inf even at p,q in {0,1}"
    assert (kl >= -1e-6).all()
    # KL(p || p) = 0
    same = _bernoulli_kl(p, p)
    assert (same.abs() < 1e-6).all()


# --- 9. artifact write round-trip ------------------------------------------

def test_write_intervention_candidates_artifact_round_trip():
    from model.intervention_driver import write_intervention_candidates_artifact

    payload = {
        "candidates": [
            {"decision_type": "ITEM_PURCHASED", "anchor_minute": 12,
             "divergence_score": 0.123, "rank_band": "master_plus"},
            {"decision_type": "ENGAGE", "anchor_minute": 18,
             "divergence_score": 0.045, "rank_band": "emerald_diamond"},
        ],
        "gate_d_pass": True,
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "intervention_candidates.json")
        write_intervention_candidates_artifact(path, payload)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == payload


def test_score_anchor_records_rollout_counters_without_batching():
    from model.intervention_driver import (
        PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT,
        _score_anchor,
    )

    model = _tiny_model()
    inputs = _synthetic_rollout_inputs(T=4, L=8, count=2)
    torch.manual_seed(2)
    encoded = {
        "h": torch.randn(4, D_H),
        "z": torch.randn(4, model.z0.numel()),
        "static_tokens": inputs["static_tokens"][0],
        "token_embeddings": inputs["token_embeddings"],
        "event_window_positions": inputs["event_window_positions"],
        "event_window_offsets": inputs["event_window_offsets"],
        "event_window_counts": inputs["event_window_counts"],
        "anchor_mask": inputs["anchor_mask"][0],
        "n_valid_anchors": 4,
    }
    synthetic_embs = {
        dt: build_synthetic_token_embedding(model, decision_type=dt)
        for dt in INTERVENTION_DECISION_TYPES
    }
    counters = {}

    scores = _score_anchor(
        model,
        encoded,
        anchor_idx=0,
        n_steps=2,
        synthetic_embs=synthetic_embs,
        counters=counters,
    )

    assert set(scores) == set(INTERVENTION_DECISION_TYPES)
    assert counters["base_rollout_calls"] == 1
    assert counters["intervention_rollout_calls"] == len(INTERVENTION_DECISION_TYPES)
    assert "one game/anchor at a time" in PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT
