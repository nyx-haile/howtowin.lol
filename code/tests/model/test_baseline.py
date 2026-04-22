import torch

from model.baseline import CausalTransformerBaseline
from model.player_features import PLAYER_FEATURE_DIM
from model.static_features import STATIC_VECTOR_DIM
from model.tokens import NUM_EVENT_TYPES


def test_forward_shape():
    model = CausalTransformerBaseline(max_puuids=100)
    B, L = 2, 30
    batch = {
        "static": torch.zeros(B, STATIC_VECTOR_DIM),
        "players": torch.randn(B, 10, PLAYER_FEATURE_DIM),
        "player_ids": torch.zeros(B, 10, dtype=torch.long),
        "tokens": torch.zeros(B, L, dtype=torch.long),
        "token_actors": torch.zeros(B, L, dtype=torch.long),
        "token_timestamps": torch.zeros(B, L),
        "key_pad_mask": torch.zeros(B, L, dtype=torch.bool),
    }
    logits = model(batch)
    assert logits.shape == (B, L, NUM_EVENT_TYPES)


def test_forward_is_differentiable():
    model = CausalTransformerBaseline(max_puuids=100)
    B, L = 2, 30
    batch = {
        "static": torch.zeros(B, STATIC_VECTOR_DIM),
        "players": torch.randn(B, 10, PLAYER_FEATURE_DIM),
        "player_ids": torch.zeros(B, 10, dtype=torch.long),
        "tokens": torch.zeros(B, L, dtype=torch.long),
        "token_actors": torch.zeros(B, L, dtype=torch.long),
        "token_timestamps": torch.zeros(B, L),
        "key_pad_mask": torch.zeros(B, L, dtype=torch.bool),
    }
    logits = model(batch)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)
