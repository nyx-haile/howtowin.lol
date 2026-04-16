import numpy as np
import pytest
from model.dataset import MatchDataset, build_puuid_index, load_split
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
