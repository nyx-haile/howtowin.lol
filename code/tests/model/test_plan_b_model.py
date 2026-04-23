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


def test_gather_event_window_handles_variable_counts_without_python_loop():
    model = PlanBModel(max_puuids=8)
    token_emb = torch.arange(2 * 6 * D_MODEL, dtype=torch.float32).view(2, 6, D_MODEL)
    batch = {
        "event_window_offsets": torch.tensor([[0, 2], [3, 4]], dtype=torch.long),
        "event_window_counts": torch.tensor([[2, 1], [1, 0]], dtype=torch.long),
        "event_window_positions": torch.tensor([1, 3, 4, 2], dtype=torch.long),
    }

    events0, mask0 = model._gather_event_window(token_emb, batch, 0)
    assert mask0.tolist() == [[True, True], [True, False]]
    assert torch.equal(events0[0, 0], token_emb[0, 1])
    assert torch.equal(events0[0, 1], token_emb[0, 3])
    assert torch.equal(events0[1, 0], token_emb[1, 2])

    events1, mask1 = model._gather_event_window(token_emb, batch, 1)
    assert mask1.tolist() == [[True], [False]]
    assert torch.equal(events1[0, 0], token_emb[0, 4])


def test_advance_event_window_matches_stepwise_reference():
    torch.manual_seed(0)
    model = PlanBModel(max_puuids=8)
    B, L = 3, 5
    h = torch.randn(B, D_H)
    z = torch.randn(B, 32)
    event_window_emb = torch.randn(B, L, D_MODEL)
    event_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, False, False, False], [False, False, False, False, False]]
    )
    static_tokens = torch.randn(B, STATIC_TOKEN_COUNT, D_MODEL)
    static_key, static_value = model._prepare_static_attention(static_tokens)

    h_scan = model._advance_event_window(h, z, event_window_emb, event_mask, static_key, static_value)

    static_ctx = model._static_event_context(z, event_window_emb, static_key, static_value)
    conditioned_action = event_window_emb + model.static_to_action(static_ctx)
    h_step = h.clone()
    for step_idx in range(L):
        valid = event_mask[:, step_idx].unsqueeze(-1)
        h_candidate = model.rssm.step(h_step, z, conditioned_action[:, step_idx])
        h_step = torch.where(valid, h_candidate, h_step)

    assert torch.allclose(h_scan, h_step, atol=1e-5, rtol=1e-5)
