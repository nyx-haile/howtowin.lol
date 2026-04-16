import torch
import pytest
from model.plan_b_model import PlanBModel
from model.dataset import MatchDataset, build_puuid_index, collate_games
from model.plan_b_eval import (
    outcome_auc_by_minute, imagination_rollout_top5, frozen_minute_0_auc,
)


@pytest.fixture(scope="module")
def tiny_model_and_ds(fixture_match_id):
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    model = PlanBModel(max_puuids=len(idx) + 1)
    return model, ds


def test_outcome_auc_by_minute_returns_per_minute_dict(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    result = outcome_auc_by_minute(model, ds, minutes=(5, 10, 15))
    assert set(result.keys()) >= {5, 10, 15}
    for m, auc in result.items():
        assert 0.0 <= auc <= 1.0


def test_imagination_rollout_top5_returns_per_step(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    res = imagination_rollout_top5(model, ds, n_steps=3)
    assert len(res) == 3
    for top5 in res:
        assert 0.0 <= top5 <= 1.0


def test_frozen_minute_0_auc_returns_scalar(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    auc = frozen_minute_0_auc(model, ds, target_minute=15)
    assert 0.0 <= auc <= 1.0
