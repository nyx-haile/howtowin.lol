import torch
from model.plan_b_model import PlanBModel
from model.tokens import ANCHOR_TOKEN


def _segment_for_test(batch, d_model=256):
    """Create placeholder anchor-windowing tensors for testing the model
    before the dataset (Task 9) produces them."""
    tokens = batch["tokens"]
    B, L = tokens.shape
    anchor_mask = (tokens == ANCHOR_TOKEN)
    anchor_positions = torch.where(anchor_mask[0])[0]
    T = anchor_positions.numel()
    max_W = 128
    batch["anchor_positions"] = anchor_positions.unsqueeze(0).expand(B, -1)
    batch["event_window_embeddings_raw"] = torch.zeros(B, T, max_W, dtype=torch.long)
    batch["window_mask"] = torch.zeros(B, T, max_W)
    batch["frame_features"] = torch.zeros(B, T, 10, 6)


def test_forward_pass_shape(fixture_match_id):
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    _segment_for_test(batch)

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
    _segment_for_test(batch)

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
