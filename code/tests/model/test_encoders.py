import torch
from model.encoders import (
    StaticContextEncoder, PlayerModelEncoder, DynamicStreamEmbedder,
    D_MODEL,
)
from model.patch_params import PATCH_VECTOR_DIM
from model.player_features import PLAYER_FEATURE_DIM
from model.tokens import VOCAB_SIZE, NUM_SLOTS


def test_static_encoder_output_shape():
    enc = StaticContextEncoder()
    x = torch.randn(4, PATCH_VECTOR_DIM)
    out = enc(x)
    assert out.shape == (4, D_MODEL)


def test_player_encoder_output_shape():
    enc = PlayerModelEncoder(max_puuids=100)
    crafted = torch.randn(4, 10, PLAYER_FEATURE_DIM)
    puuid_ids = torch.zeros(4, 10, dtype=torch.long)  # all "unknown"
    out = enc(crafted, puuid_ids)
    assert out.shape == (4, 10, D_MODEL)


def test_dynamic_embedder_output_shape():
    emb = DynamicStreamEmbedder()
    tokens = torch.zeros(4, 50, dtype=torch.long)
    actors = torch.zeros(4, 50, dtype=torch.long)
    ts = torch.zeros(4, 50)
    player_emb = torch.randn(4, 10, D_MODEL)
    out = emb(tokens, actors, ts, player_emb)
    assert out.shape == (4, 50, D_MODEL)


def test_player_residual_updates_from_known_puuid():
    enc = PlayerModelEncoder(max_puuids=10)
    crafted = torch.randn(1, 1, PLAYER_FEATURE_DIM)
    id_zero = torch.zeros(1, 1, dtype=torch.long)  # unknown (index 0)
    id_one = torch.ones(1, 1, dtype=torch.long)   # known slot 1
    out_zero = enc(crafted, id_zero)
    out_one = enc(crafted, id_one)
    # Different residual -> different output.
    assert not torch.allclose(out_zero, out_one, atol=1e-6)
