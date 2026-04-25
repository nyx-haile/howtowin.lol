"""Tests for Step 3 / Gate C skill-aware retrieval eval (m4_eval additions)."""
import json
import torch

from model.m4_eval import (
    run_skill_aware_eval,
    write_retrieval_eval_artifact,
)
from model.retrieval import (
    IndexBundle, KEY_DIM, SCHEMA_VERSION, Whitener,
)


def test_write_retrieval_eval_artifact_round_trip(tmp_path):
    path = tmp_path / "retrieval_eval.json"
    metrics = {
        "gate_c_pass": True,
        "median_effective_k": 64.0,
        "skill_aware_entropy_at_64": 0.88,
        "unrestricted_entropy_at_64": 0.87,
        "auc_at_15": 0.72,
        "nested": {"widening": {"same": 100, "adjacent": 20, "unrestricted": 5}},
    }
    write_retrieval_eval_artifact(str(path), metrics)
    with open(path) as f:
        data = json.load(f)
    assert data["gate_c_pass"] is True
    assert data["median_effective_k"] == 64.0
    assert data["nested"]["widening"]["adjacent"] == 20


def test_run_skill_aware_eval_empty_holdout_returns_sentinel_metrics():
    """With an empty holdout we should still return a well-shaped dict flagged
    as gate_c_pass=False, without exploding on nan arithmetic."""
    torch.manual_seed(0)
    N = 20
    corpus = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus), whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.zeros(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
        row_rank_band=torch.zeros(N, dtype=torch.int8),
        schema_version=SCHEMA_VERSION,
    )

    class DummyModel:
        def __call__(self, batch):
            raise AssertionError("should not be called for empty holdout")

    out = run_skill_aware_eval(
        model=DummyModel(),
        model_bundle=bundle,
        holdout_match_ids=[],
        puuid_index={},
        exclude_match_ids=set(),
        k=64,
        device="cpu",
    )
    assert out["n_queries"] == 0
    assert out["gate_c_pass"] is False
    assert out["widening_stage_counts"] == {
        "same": 0, "adjacent": 0, "unrestricted": 0,
    }
