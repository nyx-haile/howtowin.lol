import numpy as np
import torch
import pytest

from model.dataset import FRAME_FEAT_DIM, MACRO_FEAT_DIM, collate_games
from model.plan_b_eval import (
    PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT,
    outcome_auc_by_minute, imagination_rollout_top5, frozen_minute_0_auc,
)
from model.plan_b_model import PlanBModel
from model.player_features import PLAYER_FEATURE_DIM
from model.static_features import STATIC_VECTOR_DIM
from model.tokens import EVENT_TYPE_TO_ID


def _synthetic_sample(*, outcome: float = 1.0, n_anchors: int = 6, n_tokens: int = 8):
    token_ids = list(EVENT_TYPE_TO_ID.values())
    tokens = torch.tensor(
        [token_ids[i % len(token_ids)] for i in range(n_tokens)],
        dtype=torch.long,
    )
    anchor_positions = np.arange(n_anchors, dtype=np.int64)
    window_positions = [
        np.array([min(i, n_tokens - 1)], dtype=np.int64)
        for i in range(n_anchors)
    ]
    return {
        "static": torch.zeros(STATIC_VECTOR_DIM, dtype=torch.float32),
        "players": torch.zeros(10, PLAYER_FEATURE_DIM, dtype=torch.float32),
        "player_ids": torch.zeros(10, dtype=torch.long),
        "tokens": tokens,
        "token_actors": torch.zeros(n_tokens, dtype=torch.long),
        "token_targets": torch.zeros(n_tokens, dtype=torch.long),
        "token_timestamps": torch.arange(n_tokens, dtype=torch.float32) * 60000.0,
        "token_item_ids": torch.zeros(n_tokens, dtype=torch.long),
        "token_skill_slots": torch.zeros(n_tokens, dtype=torch.long),
        "token_monster_types": torch.zeros(n_tokens, dtype=torch.long),
        "token_monster_subtypes": torch.zeros(n_tokens, dtype=torch.long),
        "token_building_types": torch.zeros(n_tokens, dtype=torch.long),
        "token_lane_types": torch.zeros(n_tokens, dtype=torch.long),
        "token_tower_types": torch.zeros(n_tokens, dtype=torch.long),
        "token_ward_types": torch.zeros(n_tokens, dtype=torch.long),
        "anchor_positions": anchor_positions,
        "frame_features": np.zeros((n_anchors, 10, FRAME_FEAT_DIM), dtype=np.float32),
        "anchor_macro_features": np.zeros((n_anchors, MACRO_FEAT_DIM), dtype=np.float32),
        "decision_labels": np.zeros((n_anchors, 10), dtype=np.int64),
        "next_event_type_labels": np.ones(n_anchors, dtype=np.int64),
        "next_event_actor_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_target_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_item_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_skill_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_monster_type_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_monster_subtype_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_building_type_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_lane_type_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_tower_type_labels": np.zeros(n_anchors, dtype=np.int64),
        "next_event_ward_type_labels": np.zeros(n_anchors, dtype=np.int64),
        "window_positions": window_positions,
        "outcome": outcome,
    }


class _EvalModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, batch):
        bsz, n_anchors = batch["anchor_mask"].shape
        logits = torch.linspace(-1.0, 1.0, n_anchors).repeat(bsz, 1)
        return {
            "outcome_logits": logits,
            "anchor_mask": batch["anchor_mask"],
            "h": torch.zeros(bsz, n_anchors, 4),
            "z": torch.zeros(bsz, n_anchors, 4),
            "static_tokens": torch.zeros(bsz, 1, 4),
            "token_embeddings": torch.zeros(bsz, batch["tokens"].shape[1], 4),
        }

    def rollout_prior_single(self, **kwargs):
        return [{"anchor_repr": torch.zeros(1, 4)} for _ in range(kwargs["n_steps"])]

    def decode_anchor_repr(self, anchor_repr):
        return {"event_factors": {"type_logits": torch.arange(12, dtype=torch.float32).unsqueeze(0)}}


@pytest.fixture()
def tiny_model_and_ds():
    return _EvalModel(), [_synthetic_sample(outcome=0.0), _synthetic_sample(outcome=1.0)]


def test_outcome_auc_by_minute_returns_per_minute_dict(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    result = outcome_auc_by_minute(model, ds, minutes=(5, 10, 15))
    assert set(result.keys()) >= {5, 10, 15}
    for m, auc in result.items():
        assert 0.0 <= auc <= 1.0


def test_outcome_auc_by_minute_records_single_game_forward_gate(tiny_model_and_ds):
    model, ds = tiny_model_and_ds
    counters = {}
    outcome_auc_by_minute(model, ds, minutes=(5, 10, 15), counters=counters)

    assert counters["model_forward_calls"] == len(ds)
    assert counters["max_model_forward_batch_size"] == 1
    assert counters["stochastic_planb_forward_batching"] == "disabled_batch_size_1"
    assert "samples RSSM latents" in counters["stochastic_planb_forward_caveat"]


def test_plan_b_forward_remains_stochastic_in_eval_mode(tiny_model_and_ds):
    batch = collate_games([_synthetic_sample(n_anchors=3, n_tokens=4)])
    model = PlanBModel(max_puuids=4)
    model.eval()

    torch.manual_seed(123)
    first = model(batch)
    second = model(batch)

    valid = batch["anchor_mask"].bool().unsqueeze(-1).expand_as(first["z"])
    assert valid.any()
    assert not torch.allclose(first["z"][valid], second["z"][valid])
    assert "batch_size=1" in PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT


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
