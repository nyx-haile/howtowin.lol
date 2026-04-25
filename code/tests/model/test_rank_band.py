import torch

from model.rank_band import (
    BAND_NAMES,
    N_BANDS,
    band_distance,
    game_band,
    player_band,
    tier_to_band_tensor,
)


def test_tier_to_band_tensor_full_mapping():
    t = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
    b = tier_to_band_tensor(t)
    assert b.tolist() == [-1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
    assert len(BAND_NAMES) == N_BANDS


def test_player_band_broadcasts_over_leading_dims():
    players = torch.zeros(2, 10, 16)
    players[0, :, 0] = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:10], dtype=torch.float32,
    )
    players[1, :, 0] = 8.0  # all master
    bands = player_band(players)
    assert bands.shape == (2, 10)
    assert bands[0, 0].item() == -1
    assert bands[0, 9].item() == 3
    assert (bands[1] == 3).all().item()


def test_game_band_mode_of_ranked_players():
    players = torch.zeros(1, 10, 16)
    players[0, 0:3, 0] = 6.0  # emerald -> band 2
    players[0, 3:5, 0] = 4.0  # gold -> band 1
    assert game_band(players)[0].item() == 2


def test_game_band_all_unranked_returns_minus_one():
    players = torch.zeros(1, 10, 16)
    assert game_band(players)[0].item() == -1


def test_band_distance_same_band_is_zero():
    a = torch.tensor([0, 1, 2, 3])
    b = torch.tensor([0, 1, 2, 3])
    assert band_distance(a, b).tolist() == [0, 0, 0, 0]


def test_band_distance_cross_band_is_absolute_diff():
    a = torch.tensor([0, 0, 3])
    b = torch.tensor([1, 3, 0])
    assert band_distance(a, b).tolist() == [1, 3, 3]


def test_band_distance_unranked_is_sentinel():
    a = torch.tensor([0, -1, -1])
    b = torch.tensor([-1, 0, -1])
    # Both -1 or either -1 => sentinel N_BANDS=4
    assert band_distance(a, b).tolist() == [N_BANDS, N_BANDS, N_BANDS]
