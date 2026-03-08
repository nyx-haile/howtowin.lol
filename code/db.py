import sqlite3
import os
import time

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'howtowin.db')


def get_conn():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            match_id TEXT PRIMARY KEY,
            patch TEXT,
            queue_id INTEGER,
            game_duration_s INTEGER,
            winning_team INTEGER,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS frames (
            match_id TEXT,
            participant_slot INTEGER,
            puuid TEXT,
            team_id INTEGER,
            role TEXT,
            timestamp_ms INTEGER,
            current_gold INTEGER,
            total_gold INTEGER,
            xp INTEGER,
            level INTEGER,
            cs INTEGER,
            pos_x INTEGER,
            pos_y INTEGER,
            kills INTEGER,
            deaths INTEGER,
            assists INTEGER,
            vision_score INTEGER,
            item_ids TEXT,
            PRIMARY KEY (match_id, participant_slot, timestamp_ms)
        );

        CREATE TABLE IF NOT EXISTS players (
            puuid TEXT PRIMARY KEY,
            riot_id TEXT,
            rank_tier TEXT,
            rank_division TEXT,
            lp INTEGER,
            last_updated INTEGER
        );

        CREATE TABLE IF NOT EXISTS frame_normalized (
            match_id TEXT,
            participant_slot INTEGER,
            puuid TEXT,
            team_id INTEGER,
            role TEXT,
            timestamp_ms INTEGER,
            minute REAL,
            cs_per_min REAL,
            gold_per_min REAL,
            xp_per_min REAL,
            gold_diff_vs_opponent REAL,
            xp_diff_vs_opponent REAL,
            cs_diff_vs_opponent REAL,
            team_gold_lead REAL,
            team_level_lead REAL,
            current_gold INTEGER,
            total_gold INTEGER,
            xp INTEGER,
            level INTEGER,
            cs INTEGER,
            kills INTEGER,
            deaths INTEGER,
            assists INTEGER,
            vision_score INTEGER,
            win INTEGER,
            PRIMARY KEY (match_id, participant_slot, timestamp_ms)
        );
    """)
    conn.commit()
    conn.close()


def insert_game(match_id, patch, queue_id, game_duration_s, winning_team, created_at):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO games VALUES (?,?,?,?,?,?)",
        (match_id, patch, queue_id, game_duration_s, winning_team, created_at)
    )
    conn.commit()
    conn.close()


def insert_frames(rows):
    if not rows:
        return
    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r['match_id'], r['participant_slot'], r['puuid'], r['team_id'], r['role'],
                r['timestamp_ms'], r['current_gold'], r['total_gold'], r['xp'], r['level'],
                r['cs'], r['pos_x'], r['pos_y'], r['kills'], r['deaths'], r['assists'],
                r['vision_score'], r['item_ids'],
            )
            for r in rows
        ]
    )
    conn.commit()
    conn.close()


def insert_players(players):
    if not players:
        return
    conn = get_conn()
    now = int(time.time())
    conn.executemany(
        "INSERT OR IGNORE INTO players VALUES (?,?,?,?,?,?)",
        [
            (
                p['puuid'], p.get('riot_id', ''), p.get('rank_tier', ''),
                p.get('rank_division', ''), p.get('lp', 0), now,
            )
            for p in players
        ]
    )
    conn.commit()
    conn.close()


def upsert_player(puuid, riot_id=None, rank_tier=None, rank_division=None, lp=None):
    conn = get_conn()
    now = int(time.time())
    conn.execute(
        """INSERT INTO players (puuid, riot_id, rank_tier, rank_division, lp, last_updated)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(puuid) DO UPDATE SET
               riot_id = COALESCE(excluded.riot_id, riot_id),
               rank_tier = COALESCE(excluded.rank_tier, rank_tier),
               rank_division = COALESCE(excluded.rank_division, rank_division),
               lp = COALESCE(excluded.lp, lp),
               last_updated = excluded.last_updated""",
        (puuid, riot_id, rank_tier, rank_division, lp, now)
    )
    conn.commit()
    conn.close()


def insert_frame_normalized(rows):
    if not rows:
        return
    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO frame_normalized VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r['match_id'], r['participant_slot'], r['puuid'], r['team_id'], r['role'],
                r['timestamp_ms'], r['minute'],
                r['cs_per_min'], r['gold_per_min'], r['xp_per_min'],
                r['gold_diff_vs_opponent'], r['xp_diff_vs_opponent'], r['cs_diff_vs_opponent'],
                r['team_gold_lead'], r['team_level_lead'],
                r['current_gold'], r['total_gold'], r['xp'], r['level'], r['cs'],
                r['kills'], r['deaths'], r['assists'], r['vision_score'],
                r['win'],
            )
            for r in rows
        ]
    )
    conn.commit()
    conn.close()


def game_exists(match_id):
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM games WHERE match_id = ?", (match_id,)).fetchone()
    conn.close()
    return row is not None


# Initialize schema on import
init()
