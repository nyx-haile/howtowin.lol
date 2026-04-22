import os
import sys
import sqlite3
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import db
from model.dataset import OBJECTIVE_EVENT_TYPES_SQL
from model.player_match_stats import (
    _covered_match_count,
    _source_match_count,
    ensure_player_match_stats,
)
from model.tokenizer import MEANINGFUL_EVENT_TYPES_SQL

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
            'key_moments', 'player_match_stats', 'player_progress', 'players'
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


def test_player_match_stats_materializes_from_frames():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'test.db')
        db.init_db(test_path)
        conn = db.get_conn(test_path)
        db.insert_game(conn, 'NA1_123', '14.15', 420, 1800, 100, 1700000000)
        db.insert_frame(conn, 'NA1_123', 1, 'puuid_abc', 100, 'TOP', 540000,
                         current_gold=500, total_gold=500, xp=600, level=2,
                         cs=12, jungle_cs=0, pos_x=1200, pos_y=5000,
                         kills=1, deaths=0, assists=2,
                         ward_count=1, item_ids=[1055])
        db.insert_frame(conn, 'NA1_123', 1, 'puuid_abc', 100, 'TOP', 600000,
                         current_gold=800, total_gold=900, xp=1100, level=4,
                         cs=24, jungle_cs=0, pos_x=1500, pos_y=5200,
                         kills=2, deaths=1, assists=3,
                         ward_count=2, item_ids=[1055, 2003])
        conn.commit()

        ensure_player_match_stats(conn)
        row = conn.execute(
            "SELECT team_id, role, created_at, winning_team, window_10_rows, cs_10_sum, total_gold_10_sum "
            "FROM player_match_stats WHERE puuid = 'puuid_abc' AND match_id = 'NA1_123'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row['team_id'] == 100
        assert row['role'] == 'TOP'
        assert row['created_at'] == 1700000000
        assert row['winning_team'] == 100
        assert row['window_10_rows'] == 2
        assert row['cs_10_sum'] == 36.0
        assert row['total_gold_10_sum'] == 1400.0


def test_player_match_stats_source_count_ignores_games_without_frames():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'test.db')
        db.init_db(test_path)
        conn = db.get_conn(test_path)
        db.insert_game(conn, 'NA1_frameful', '14.15', 420, 1800, 100, 1700000000)
        db.insert_game(conn, 'NA1_frameless', '14.15', 420, 1800, 100, 1700000001)
        db.insert_frame(conn, 'NA1_frameful', 1, 'puuid_abc', 100, 'TOP', 600000,
                         current_gold=500, total_gold=500, xp=600, level=2,
                         cs=12, jungle_cs=0, pos_x=1200, pos_y=5000,
                         kills=1, deaths=0, assists=2,
                         ward_count=1, item_ids=[1055])
        conn.commit()

        assert _source_match_count(conn) == 1
        ensure_player_match_stats(conn)
        assert _covered_match_count(conn) == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM player_match_stats WHERE match_id = 'NA1_frameless'"
        ).fetchone()[0] == 0
        conn.close()


def test_event_queries_use_match_timestamp_index_without_temp_sort():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'test.db')
        db.init_db(test_path)
        conn = db.get_conn(test_path)
        db.insert_event(conn, 'NA1_123', 600000, 'CHAMPION_KILL', killer_id=2, victim_id=3)
        db.insert_event(conn, 'NA1_123', 660000, 'ELITE_MONSTER_KILL', killer_id=2, killer_team=100)
        db.insert_event(conn, 'NA1_123', 720000, 'ITEM_PURCHASED', participant_id=2)
        conn.commit()

        meaningful_placeholders = ",".join("?" for _ in MEANINGFUL_EVENT_TYPES_SQL)
        meaningful_plan = conn.execute(
            f"""EXPLAIN QUERY PLAN
                SELECT timestamp_ms, event_type
                FROM events
                WHERE match_id = ? AND event_type IN ({meaningful_placeholders})
                ORDER BY timestamp_ms""",
            ('NA1_123', *MEANINGFUL_EVENT_TYPES_SQL),
        ).fetchall()

        objective_placeholders = ",".join("?" for _ in OBJECTIVE_EVENT_TYPES_SQL)
        objective_plan = conn.execute(
            f"""EXPLAIN QUERY PLAN
                SELECT timestamp_ms, event_type
                FROM events
                WHERE match_id = ? AND event_type IN ({objective_placeholders})
                ORDER BY timestamp_ms""",
            ('NA1_123', *OBJECTIVE_EVENT_TYPES_SQL),
        ).fetchall()
        conn.close()

        meaningful_details = [row['detail'] for row in meaningful_plan]
        objective_details = [row['detail'] for row in objective_plan]

        assert any('idx_events_match_ts' in d or 'idx_events_meaningful_match_ts' in d for d in meaningful_details)
        assert any('idx_events_match_ts' in d or 'idx_events_objective_match_ts' in d for d in objective_details)
        assert all('USE TEMP B-TREE' not in d for d in meaningful_details)
        assert all('USE TEMP B-TREE' not in d for d in objective_details)
