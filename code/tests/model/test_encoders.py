import torch

from model.encoders import D_MODEL, DynamicStreamEmbedder, PlayerModelEncoder, StaticContextEncoder
from model.player_features import PLAYER_FEATURE_DIM
from model.static_features import STATIC_TOKEN_COUNT, STATIC_VECTOR_DIM


def test_static_encoder_output_shape():
    enc = StaticContextEncoder()
    x = torch.zeros(4, STATIC_VECTOR_DIM)
    x[:, 0] = 266  # champ id
    x[:, 10] = 1   # side flag
    x[:, 20] = 420 # queue
    out = enc(x)
    assert out.shape == (4, STATIC_TOKEN_COUNT, D_MODEL)


def test_player_encoder_output_shape():
    enc = PlayerModelEncoder(max_puuids=100)
    crafted = torch.randn(4, 10, PLAYER_FEATURE_DIM)
    puuid_ids = torch.zeros(4, 10, dtype=torch.long)
    out = enc(crafted, puuid_ids)
    assert out.shape == (4, 10, D_MODEL)


def test_dynamic_embedder_output_shape_with_payloads():
    emb = DynamicStreamEmbedder()
    tokens = torch.zeros(4, 50, dtype=torch.long)
    actors = torch.zeros(4, 50, dtype=torch.long)
    targets = torch.zeros(4, 50, dtype=torch.long)
    ts = torch.zeros(4, 50)
    player_emb = torch.randn(4, 10, D_MODEL)
    out = emb(tokens, actors, ts, player_emb, targets=targets)
    assert out.shape == (4, 50, D_MODEL)


def test_player_residual_updates_from_known_puuid():
    enc = PlayerModelEncoder(max_puuids=10)
    crafted = torch.randn(1, 1, PLAYER_FEATURE_DIM)
    id_zero = torch.zeros(1, 1, dtype=torch.long)
    id_one = torch.ones(1, 1, dtype=torch.long)
    out_zero = enc(crafted, id_zero)
    out_one = enc(crafted, id_one)
    assert not torch.allclose(out_zero, out_one, atol=1e-6)
