import torch
from model.plan_b_model import PlanBModel


def test_forward_pass_shape(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])

    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)

    T = out["n_anchors"]
    assert out["event_logits"].shape == (1, T, 11)
    assert out["outcome_logits"].shape == (1, T)
    assert out["decision_logits"].shape == (1, T, 10, 6)
    assert out["frame_mu"].shape == (1, T, 10, 6)
    assert out["frame_logvar"].shape == (1, T, 10, 6)
    assert out["post_mu"].shape == (1, T, 32)
    assert out["prior_mu"].shape == (1, T, 32)


def test_model_forward_is_differentiable(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])

    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)
    loss = (
        out["event_logits"].sum() + out["outcome_logits"].sum()
        + out["decision_logits"].sum() + out["frame_mu"].sum()
    )
    loss.backward()
    grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.parameters())
    assert grad_found


def test_forward_returns_per_anchor_h(fixture_match_id):
    from model.plan_b_model import PlanBModel, D_H
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)
    T = out["n_anchors"]
    assert "h" in out, "PlanBModel.forward must return per-anchor h"
    assert out["h"].shape == (1, T, D_H), \
        f"expected (1, {T}, {D_H}), got {tuple(out['h'].shape)}"
