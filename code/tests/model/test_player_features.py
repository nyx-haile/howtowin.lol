import numpy as np
from model.player_features import (
    player_feature_vector, PLAYER_FEATURE_DIM, RANK_TIER_ORDER,
)


def test_feature_shape_for_puuid_in_corpus(all_match_ids, fixture_match_id):
    from raw_db import get_raw_match
    match, _ = get_raw_match(fixture_match_id)
    puuid = match["info"]["participants"][0]["puuid"]
    vec = player_feature_vector(puuid)
    assert vec.shape == (PLAYER_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_unknown_puuid_returns_zero_vector():
    vec = player_feature_vector("nonexistent-puuid-xxx")
    assert vec.shape == (PLAYER_FEATURE_DIM,)
    assert np.allclose(vec, 0.0)


def test_rank_tier_order_monotonic():
    # CHALLENGER > GRANDMASTER > MASTER > DIAMOND > ...
    assert RANK_TIER_ORDER["CHALLENGER"] > RANK_TIER_ORDER["GRANDMASTER"]
    assert RANK_TIER_ORDER["GRANDMASTER"] > RANK_TIER_ORDER["MASTER"]
    assert RANK_TIER_ORDER["DIAMOND"] > RANK_TIER_ORDER["EMERALD"]
