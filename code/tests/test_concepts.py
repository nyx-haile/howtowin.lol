import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import db
import features
import concepts


def _setup_corpus(db_path):
    """Create multiple test matches to train on."""
    db.init_db(db_path)
    conn = db.get_conn(db_path)

    # Create 30 fake matches where winning team has more gold/kills/dragons
    for i in range(30):
        mid = f'TEST_{i}'
        winning = 100 if i % 3 != 0 else 200
        db.insert_game(conn, mid, '14.15', 420, 1800, winning, 1700000000 + i)

        for slot in range(1, 11):
            team = 100 if slot <= 5 else 200
            role = ['TOP', 'JGL', 'MID', 'BOT', 'SUP'][(slot - 1) % 5]
            is_winning = (team == winning)
            gold = 5000 + (1000 if is_winning else 0) + (i * 50)
            wards = 8 + (4 if is_winning else 0)
            db.insert_frame(conn, mid, slot, f'p_{i}_{slot}', team, role, 600000,
                             current_gold=1000, total_gold=gold, xp=3000, level=6,
                             cs=80, jungle_cs=10, pos_x=7000, pos_y=7000,
                             kills=3 if is_winning else 1,
                             deaths=1 if is_winning else 3,
                             assists=2, ward_count=wards, item_ids=[])

        # Winning team gets dragon
        db.insert_event(conn, mid, 300000, 'ELITE_MONSTER_KILL',
                         killer_id=2 if winning == 100 else 7,
                         killer_team=winning,
                         details='{"monsterType": "DRAGON", "monsterSubType": "WATER_DRAGON"}')

        # Winning team takes a tower (losing team loses a tower)
        lost_tower_team = 200 if winning == 100 else 100
        db.insert_event(conn, mid, 500000, 'BUILDING_KILL',
                         killer_id=1, team_id=lost_tower_team,
                         details='{"buildingType": "TOWER_BUILDING", "towerType": "OUTER_TURRET", "laneType": "MID_LANE"}')

    conn.commit()
    conn.close()


def test_discover_concepts_returns_list():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, 'test.db')
        _setup_corpus(db_path)
        found = concepts.discover_concepts(db_path)
        assert isinstance(found, list)
        assert len(found) > 0


def test_concepts_have_required_fields():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, 'test.db')
        _setup_corpus(db_path)
        found = concepts.discover_concepts(db_path)
        for c in found:
            assert 'concept_id' in c
            assert 'name' in c
            assert 'explanation' in c
            assert 'importance' in c
            assert 'feature_key' in c


def test_gold_or_kill_feature_is_discovered():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, 'test.db')
        _setup_corpus(db_path)
        found = concepts.discover_concepts(db_path)
        feature_keys = [c['feature_key'] for c in found]
        assert any('gold' in k or 'kill' in k for k in feature_keys), \
            f"Expected gold/kill feature in top concepts, got: {feature_keys}"


def test_find_key_moments():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, 'test.db')
        _setup_corpus(db_path)
        found_concepts = concepts.discover_concepts(db_path)
        moments = concepts.find_key_moments('TEST_0', found_concepts, db_path)
        assert isinstance(moments, list)
        if moments:
            m = moments[0]
            assert 'timestamp_ms' in m
            assert 'concept_id' in m
            assert 'description' in m
