import hashlib
from model.cold_holdout import (
    build_player_cold_holdout, COLD_HOLDOUT_PATH, COLD_HOLDOUT_PUUID_COUNT,
)


def test_build_is_deterministic():
    a = build_player_cold_holdout()
    b = build_player_cold_holdout()
    assert a == b


def test_uses_expected_number_of_puuids():
    result = build_player_cold_holdout()
    puuids, match_ids = result
    assert len(puuids) == COLD_HOLDOUT_PUUID_COUNT


def test_match_ids_actually_contain_those_puuids():
    from db import get_conn
    puuids, match_ids = build_player_cold_holdout()
    conn = get_conn()
    for mid in list(match_ids)[:5]:
        rows = conn.execute(
            "SELECT DISTINCT puuid FROM frames WHERE match_id = ?", (mid,)
        ).fetchall()
        participants = {r["puuid"] for r in rows}
        assert participants & puuids, f"cold match {mid} has no cold puuid"
    conn.close()


def test_no_overlap_with_game_cold_holdout():
    from model.dataset import load_split
    game_cold = set(load_split("holdout"))
    _puuids, cold_match_ids = build_player_cold_holdout()
    overlap = game_cold & cold_match_ids
    assert len(cold_match_ids) > len(overlap), (
        "player-cold holdout adds no new held-out games beyond game-cold"
    )


def test_constant_path_points_into_splits_dir():
    assert "data/splits/plan_b_cold_holdout.txt" in str(COLD_HOLDOUT_PATH)
