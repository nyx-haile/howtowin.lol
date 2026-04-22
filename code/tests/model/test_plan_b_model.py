import torch

from model.encoders import D_MODEL
from model.plan_b_model import D_H, D_R, PlanBModel
from model.static_features import STATIC_TOKEN_COUNT


def test_forward_pass_shape(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games, FRAME_FEAT_DIM

    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])

    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)

    T = out["n_anchors"]
    assert out["event_logits"].shape == (1, T, 11)
    assert out["event_factors"]["type_logits"].shape == (1, T, 12)
    assert out["outcome_logits"].shape == (1, T)
    assert out["decision_logits"].shape == (1, T, 10, 6)
    assert out["frame_mu"].shape == (1, T, 10, FRAME_FEAT_DIM)
    assert out["frame_logvar"].shape == (1, T, 10, FRAME_FEAT_DIM)
    assert out["post_mu"].shape == (1, T, 32)
    assert out["prior_mu"].shape == (1, T, 32)
    assert out["anchor_repr"].shape == (1, T, D_R)
    assert out["anchor_mask"].shape == (1, T)
    assert out["static_tokens"].shape == (1, STATIC_TOKEN_COUNT, D_MODEL)
    assert out["static_context"].shape == (1, T, D_MODEL)
    assert out["static_attention_weights"].shape == (1, T, STATIC_TOKEN_COUNT)


def test_model_forward_is_differentiable(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games

    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])

    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)
    loss = (
        out["event_logits"].sum()
        + out["outcome_logits"].sum()
        + out["decision_logits"].sum()
        + out["frame_mu"].sum()
    )
    loss.backward()
    grad_found = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grad_found


def test_forward_returns_per_anchor_h(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games

    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)
    T = out["n_anchors"]
    assert "h" in out
    assert out["h"].shape == (1, T, D_H)


def test_forward_zeroes_padded_anchor_outputs(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games

    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    base = ds[0]
    short = dict(base)
    short_T = 2
    short["anchor_positions"] = base["anchor_positions"][:short_T].copy()
    short["frame_features"] = base["frame_features"][:short_T].copy()
    short["anchor_macro_features"] = base["anchor_macro_features"][:short_T].copy()
    short["decision_labels"] = base["decision_labels"][:short_T].copy()
    short["next_event_type_labels"] = base["next_event_type_labels"][:short_T].copy()
    short["next_event_actor_labels"] = base["next_event_actor_labels"][:short_T].copy()
    short["next_event_target_labels"] = base["next_event_target_labels"][:short_T].copy()
    short["next_event_item_labels"] = base["next_event_item_labels"][:short_T].copy()
    short["next_event_skill_labels"] = base["next_event_skill_labels"][:short_T].copy()
    short["next_event_monster_type_labels"] = base["next_event_monster_type_labels"][:short_T].copy()
    short["next_event_monster_subtype_labels"] = base["next_event_monster_subtype_labels"][:short_T].copy()
    short["next_event_building_type_labels"] = base["next_event_building_type_labels"][:short_T].copy()
    short["next_event_lane_type_labels"] = base["next_event_lane_type_labels"][:short_T].copy()
    short["next_event_tower_type_labels"] = base["next_event_tower_type_labels"][:short_T].copy()
    short["next_event_ward_type_labels"] = base["next_event_ward_type_labels"][:short_T].copy()
    short["window_positions"] = base["window_positions"][:short_T]

    batch = collate_games([base, short])
    model = PlanBModel(max_puuids=len(idx) + 1)
    out = model(batch)

    assert batch["anchor_mask"][1, short_T:].sum().item() == 0
    assert torch.allclose(out["anchor_repr"][1, short_T:], torch.zeros_like(out["anchor_repr"][1, short_T:]))
