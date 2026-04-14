import os
import sys
import sqlite3
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import db

def test_init_creates_all_tables():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'test.db')
        db.init_db(test_path)
        conn = sqlite3.connect(test_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        conn.close()
        assert tables == [
            'concepts', 'events', 'frames', 'games',
            'key_moments', 'player_progress', 'players'
        ]

def test_insert_and_read_game():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'test.db')
        db.init_db(test_path)
        conn = db.get_conn(test_path)
        db.insert_game(conn, 'NA1_123', '14.15', 420, 1800, 100, 1700000000)
        conn.commit()
        row = conn.execute("SELECT * FROM games WHERE match_id = 'NA1_123'").fetchone()
        assert row['match_id'] == 'NA1_123'
        assert row['winning_team'] == 100
        conn.close()

def test_insert_event():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'test.db')
        db.init_db(test_path)
        conn = db.get_conn(test_path)
        db.insert_event(conn, 'NA1_123', 600000, 'ELITE_MONSTER_KILL',
                         killer_id=2, killer_team=100,
                         details='{"monsterType": "DRAGON", "monsterSubType": "WATER_DRAGON"}')
        conn.commit()
        row = conn.execute("SELECT * FROM events WHERE match_id = 'NA1_123'").fetchone()
        assert row['event_type'] == 'ELITE_MONSTER_KILL'
        assert row['killer_team'] == 100
        conn.close()

def test_insert_frame():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'test.db')
        db.init_db(test_path)
        conn = db.get_conn(test_path)
        db.insert_frame(conn, 'NA1_123', 1, 'puuid_abc', 100, 'TOP', 60000,
                         current_gold=500, total_gold=500, xp=600, level=2,
                         cs=12, jungle_cs=0, pos_x=1200, pos_y=5000,
                         kills=0, deaths=0, assists=0,
                         ward_count=1, item_ids=[1055])
        conn.commit()
        row = conn.execute("SELECT * FROM frames WHERE match_id = 'NA1_123'").fetchone()
        assert row['puuid'] == 'puuid_abc'
        assert row['total_gold'] == 500
        assert row['ward_count'] == 1
        conn.close()
