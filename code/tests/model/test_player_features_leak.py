import pytest
import numpy as np
from model.player_features import player_feature_vector


@pytest.fixture(scope="module")
def sample_puuid():
    from db import get_conn
    conn = get_conn()
    row = conn.execute(
        "SELECT puuid FROM frames GROUP BY puuid HAVING COUNT(DISTINCT match_id) > 2 LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None, "test requires a puuid with >2 games in frames"
    return row["puuid"]


def test_feature_vector_respects_exclude_match_ids(sample_puuid):
    vec_full = player_feature_vector(sample_puuid)

    # Find any match this puuid played in.
    from db import get_conn
    conn = get_conn()
    row = conn.execute(
        "SELECT DISTINCT match_id FROM frames WHERE puuid = ? LIMIT 1",
        (sample_puuid,),
    ).fetchone()
    conn.close()
    assert row is not None
    mid = row["match_id"]

    vec_excl = player_feature_vector(sample_puuid, exclude_match_ids={mid})
    # Feature vector MUST change when we remove a game the player participated in.
    assert not (vec_full == vec_excl).all(), (
        "player_feature_vector did not change when a participating match was excluded"
    )


def test_feature_vector_excludes_chronologically_later_games(sample_puuid):
    from db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT f.match_id, g.created_at "
        "FROM frames f JOIN games g USING(match_id) "
        "WHERE f.puuid = ? ORDER BY g.created_at ASC",
        (sample_puuid,),
    ).fetchall()
    conn.close()
    assert len(rows) >= 2, "test requires a puuid with >=2 chronologically-ordered games"

    earliest_mid = rows[0]["match_id"]

    # When we exclude the earliest game, later games should also be filtered
    # (to prevent the model seeing the player's future).
    vec_excl_earliest = player_feature_vector(
        sample_puuid, exclude_match_ids={earliest_mid}
    )
    vec_excl_all = player_feature_vector(
        sample_puuid, exclude_match_ids={r["match_id"] for r in rows}
    )
    # Excluding earliest alone should match excluding all (causal exclusion).
    np.testing.assert_allclose(vec_excl_earliest, vec_excl_all)


def test_feature_vector_no_exclusion_matches_full_corpus(sample_puuid):
    vec_a = player_feature_vector(sample_puuid)
    vec_b = player_feature_vector(sample_puuid, exclude_match_ids=set())
    np.testing.assert_allclose(vec_a, vec_b)
