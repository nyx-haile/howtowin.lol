import numpy as np
import torch
import pytest
from model.dataset import (
    MatchDataset, build_puuid_index, load_split, collate_games,
    FRAME_FEAT_DIM, NO_DECISION,
)
from model.cold_holdout import build_player_cold_holdout, save_player_cold_holdout


@pytest.fixture(scope="module")
def cold_sets():
    puuids, match_ids = build_player_cold_holdout()
    return puuids, match_ids


def test_dataset_accepts_and_uses_exclude_match_ids(fixture_match_id, cold_sets):
    _puuids, cold_ids = cold_sets
    exclude = {fixture_match_id} | cold_ids
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx, exclude_match_ids=exclude)
    sample = ds[0]
    assert "players" in sample
    assert np.all(np.isfinite(sample["players"].numpy()))


def test_dataset_without_exclude_matches_plan_a_behavior(fixture_match_id):
    idx = build_puuid_index([fixture_match_id])
    ds_a = MatchDataset([fixture_match_id], puuid_index=idx)
    ds_b = MatchDataset([fixture_match_id], puuid_index=idx, exclude_match_ids=None)
    np.testing.assert_allclose(ds_a[0]["players"].numpy(), ds_b[0]["players"].numpy())


def test_load_split_cold_returns_cold_ids():
    save_player_cold_holdout()  # ensure file exists
    cold = set(load_split("cold"))
    _puuids, expected = build_player_cold_holdout()
    assert cold == expected


def test_collate_emits_anchor_windowing_tensors(fixture_match_id):
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    B = 1
    T = batch["anchor_positions"].shape[1]

    assert batch["anchor_positions"].shape == (B, T)
    assert batch["anchor_positions"].dtype == torch.long

    assert batch["frame_features"].shape == (B, T, 10, FRAME_FEAT_DIM)
    assert batch["frame_features"].dtype == torch.float32

    assert batch["decision_labels"].shape == (B, T, 10)
    assert batch["decision_labels"].dtype == torch.long

    assert batch["event_window_embeddings_raw"].shape[0] == B
    assert batch["event_window_embeddings_raw"].shape[1] == T
    assert batch["event_window_embeddings_raw"].shape[2] == 128
    assert batch["event_window_embeddings_raw"].dtype == torch.long

    assert batch["window_mask"].shape == (B, T, 128)
    assert batch["window_mask"].dtype == torch.float32

    assert batch["outcome"].shape == (B,)
    assert batch["outcome"].dtype == torch.float32
    assert batch["outcome"][0] in (0.0, 1.0)

    # At least some window positions should be non-zero.
    assert batch["window_mask"].sum() > 0
    # At least some frame features should be non-zero (gold, xp, etc.).
    assert batch["frame_features"].abs().sum() > 0
