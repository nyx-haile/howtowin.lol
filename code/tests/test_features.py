import os
import sys
import json
import sqlite3
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import db
import features


def _setup_test_match(db_path):
    """Create a minimal match with two frames and some events for testing."""
    db.init_db(db_path)
    conn = db.get_conn(db_path)

    db.insert_game(conn, 'TEST_1', '14.15', 420, 1200, 100, 1700000000)

    # Frame at 0ms — all players at base
    for slot in range(1, 11):
        team = 100 if slot <= 5 else 200
        role = ['TOP', 'JGL', 'MID', 'BOT', 'SUP'][(slot - 1) % 5]
        db.insert_frame(conn, 'TEST_1', slot, f'puuid_{slot}', team, role, 0,
                         current_gold=500, total_gold=500, xp=0, level=1,
                         cs=0, jungle_cs=0, pos_x=500, pos_y=500,
                         kills=0, deaths=0, assists=0, ward_count=0, item_ids=[])

    # Frame at 600000ms (10 min) — team 100 ahead
    for slot in range(1, 11):
        team = 100 if slot <= 5 else 200
        role = ['TOP', 'JGL', 'MID', 'BOT', 'SUP'][(slot - 1) % 5]
        gold = 4000 if team == 100 else 3200
        db.insert_frame(conn, 'TEST_1', slot, f'puuid_{slot}', team, role, 600000,
                         current_gold=1000, total_gold=gold, xp=3000, level=6,
                         cs=80 if team == 100 else 65, jungle_cs=0,
                         pos_x=7000 if team == 100 else 8000,
                         pos_y=7000 if team == 100 else 8000,
                         kills=2 if team == 100 else 1, deaths=1, assists=1,
                         ward_count=5 if team == 100 else 2, item_ids=[1055])

    # Dragon kill by team 100 at 5 min
    db.insert_event(conn, 'TEST_1', 300000, 'ELITE_MONSTER_KILL',
                     killer_id=2, killer_team=100,
                     position_x=10352, position_y=4735,
                     details='{"monsterType": "DRAGON", "monsterSubType": "WATER_DRAGON"}')

    # Tower kill by team 200 at 8 min — team 100 lost a tower
    db.insert_event(conn, 'TEST_1', 480000, 'BUILDING_KILL',
                     killer_id=9, team_id=100,
                     position_x=10504, position_y=1029,
                     details='{"buildingType": "TOWER_BUILDING", "towerType": "OUTER_TURRET", "laneType": "BOT_LANE"}')

    conn.commit()
    conn.close()
    return db_path


def test_compute_game_state_returns_features():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _setup_test_match(os.path.join(tmp, 'test.db'))
        states = features.compute_game_states('TEST_1', db_path)
        assert len(states) == 2  # two timestamps
        state_10m = states[600000]
        assert 'team_100' in state_10m
        assert 'team_200' in state_10m
        t100 = state_10m['team_100']
        assert 'total_gold' in t100
        assert 'dragon_count' in t100
        assert 'tower_count' in t100
        assert 'ward_count' in t100


def test_compute_feature_vector():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _setup_test_match(os.path.join(tmp, 'test.db'))
        vectors = features.compute_feature_vectors('TEST_1', db_path)
        assert len(vectors) > 0
        v = vectors[0]
        assert isinstance(v, dict)
        assert 'timestamp_ms' in v
        assert 'team_gold_lead' in v
        assert 'team_dragon_diff' in v
        assert 'team_ward_diff' in v
        assert 'won' in v


def test_feature_vector_has_correct_outcome():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _setup_test_match(os.path.join(tmp, 'test.db'))
        vectors = features.compute_feature_vectors('TEST_1', db_path)
        team100_vecs = [v for v in vectors if v['team_id'] == 100]
        assert all(v['won'] == 1 for v in team100_vecs)
        team200_vecs = [v for v in vectors if v['team_id'] == 200]
        assert all(v['won'] == 0 for v in team200_vecs)
