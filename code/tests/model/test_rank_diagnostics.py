import json

import numpy as np
import torch

from model.rank_diagnostics import (
    BAND_NAMES,
    N_BANDS,
    _game_band,
    _player_band,
    _tier_to_band_tensor,
    classify_rank_use,
    fit_shallow_probe,
    write_rank_diagnosis_artifact,
)


def test_tier_to_band_mapping_covers_all_tiers():
    t = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32
    )
    bands = _tier_to_band_tensor(t)
    assert bands[0].item() == -1
    assert bands[1].item() == 0 and bands[3].item() == 0
    assert bands[4].item() == 1 and bands[5].item() == 1
    assert bands[6].item() == 2 and bands[7].item() == 2
    assert bands[8].item() == 3 and bands[10].item() == 3
    assert len(BAND_NAMES) == N_BANDS


def test_game_band_mode_of_ranked_players():
    players = torch.zeros(1, 10, 16)
    # 3 ranked players in emerald-diamond, 2 in gold-platinum, 5 unranked
    players[0, 0:3, 0] = 6.0   # emerald -> band 2
    players[0, 3:5, 0] = 4.0   # gold -> band 1
    band = _game_band(players)
    # mode over ranked players = emerald_diamond (2)
    assert band[0].item() == 2


def test_game_band_uses_single_ranked_player_as_lobby_band():
    players = torch.zeros(1, 10, 16)
    # Only 1 ranked player (threshold is 1); its band represents the lobby.
    players[0, 0, 0] = 4.0  # Gold -> gold_platinum (1)
    band = _game_band(players)
    assert band[0].item() == 1


def test_game_band_returns_unranked_when_all_unranked():
    players = torch.zeros(1, 10, 16)
    band = _game_band(players)
    assert band[0].item() == -1


def test_player_band_per_participant():
    players = torch.zeros(1, 10, 16)
    players[0, :, 0] = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], dtype=torch.float32
    )
    bands = _player_band(players)
    assert bands[0, 0].item() == -1
    assert bands[0, 1].item() == 0
    assert bands[0, 9].item() == 3


def test_fit_shallow_probe_recovers_separable_signal():
    rng = np.random.default_rng(0)
    d = 16
    centers = np.eye(N_BANDS, d)
    X = np.concatenate(
        [centers[c] + 0.05 * rng.normal(size=(200, d)) for c in range(N_BANDS)],
        axis=0,
    )
    y = np.concatenate([np.full(200, c) for c in range(N_BANDS)])
    auc = fit_shallow_probe(X, y, seed=0)
    assert auc >= 0.95


def test_fit_shallow_probe_noise_is_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 16))
    y = rng.integers(low=0, high=N_BANDS, size=800)
    auc = fit_shallow_probe(X, y, seed=0)
    assert 0.40 <= auc <= 0.60


def test_fit_shallow_probe_returns_nan_for_tiny_input():
    X = np.zeros((10, 8), dtype=np.float32)
    y = np.zeros((10,), dtype=np.int64)
    auc = fit_shallow_probe(X, y)
    assert np.isnan(auc)


def test_classify_under_use_rule():
    c = classify_rank_use(
        {
            "player_emb": 0.82,
            "h_t": 0.54,
            "post_mu": 0.55,
            "prior_mu": 0.52,
            "retrieval_key": 0.55,
        },
        swap_delta=0.01,
    )
    assert c == "under_use"


def test_classify_over_leak_rule():
    c = classify_rank_use(
        {
            "player_emb": 0.82,
            "h_t": 0.80,
            "post_mu": 0.75,
            "prior_mu": 0.72,
            "retrieval_key": 0.78,
        },
        swap_delta=0.08,
    )
    assert c == "over_leak"


def test_classify_mixed_or_unclear_rule():
    c = classify_rank_use(
        {
            "player_emb": 0.60,
            "h_t": 0.55,
            "post_mu": 0.52,
            "prior_mu": 0.51,
            "retrieval_key": 0.55,
        },
        swap_delta=0.02,
    )
    assert c == "mixed_or_unclear"


def test_over_leak_requires_swap_evidence():
    c = classify_rank_use(
        {"player_emb": 0.82, "h_t": 0.80, "post_mu": 0.75, "prior_mu": 0.72, "retrieval_key": 0.78},
        swap_delta=0.01,
    )
    assert c == "mixed_or_unclear"


def test_write_rank_diagnosis_artifact_round_trip(tmp_path):
    path = tmp_path / "rank_diagnosis.json"
    metrics = {
        "classification": "under_use",
        "probe_auc_by_layer": {"player_emb": 0.80, "h_t": 0.54},
        "swap_delta": 0.01,
        "gate_b_pass": True,
        "details": {"k": "v", "nested": {"n": 1}},
    }
    write_rank_diagnosis_artifact(str(path), metrics)
    with open(path) as f:
        data = json.load(f)
    assert data["classification"] == "under_use"
    assert data["gate_b_pass"] is True
    assert data["probe_auc_by_layer"]["h_t"] == 0.54
