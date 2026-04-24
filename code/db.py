import sqlite3
import json
import os
import time

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'howtowin.db')


def _resolve_db_path(db_path=None):
    if db_path:
        return db_path
    return os.environ.get('HOWL_DB_PATH', DEFAULT_DB_PATH)


def get_conn(db_path=None):
    conn = sqlite3.connect(_resolve_db_path(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")      # WAL-safe, skip redundant fsyncs
    conn.execute("PRAGMA busy_timeout=30000")      # 30s wait instead of locking error
    conn.execute("PRAGMA cache_size=-65536")       # 64MB page cache per connection
    conn.execute("PRAGMA temp_store=MEMORY")       # keep temp b-trees in RAM
    conn.execute("PRAGMA mmap_size=8589934592")    # 8GB memory-map window
    conn.execute("PRAGMA wal_autocheckpoint=4000") # 4000 pages (~16MB) between checkpoints
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path=None):
    conn = get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            match_id TEXT PRIMARY KEY,
            patch TEXT,
            queue_id INTEGER,
            game_duration_s INTEGER,
            winning_team INTEGER,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS players (
            puuid TEXT PRIMARY KEY,
            riot_id TEXT,
            rank_tier TEXT,
            rank_division TEXT,
            lp INTEGER,
            last_updated INTEGER
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
            jungle_cs INTEGER,
            pos_x INTEGER,
            pos_y INTEGER,
            kills INTEGER DEFAULT 0,
            deaths INTEGER DEFAULT 0,
            assists INTEGER DEFAULT 0,
            ward_count INTEGER DEFAULT 0,
            item_ids TEXT DEFAULT '[]',
            PRIMARY KEY (match_id, participant_slot, timestamp_ms)
        );

        CREATE TABLE IF NOT EXISTS events (
            match_id TEXT,
            timestamp_ms INTEGER,
            event_type TEXT,
            participant_id INTEGER,
            killer_id INTEGER,
            victim_id INTEGER,
            killer_team INTEGER,
            team_id INTEGER,
            position_x INTEGER,
            position_y INTEGER,
            details TEXT DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(match_id, event_type);
        CREATE INDEX IF NOT EXISTS idx_events_match_ts ON events(match_id, timestamp_ms);
        CREATE INDEX IF NOT EXISTS idx_events_meaningful_match_ts
            ON events(match_id, timestamp_ms)
            WHERE event_type IN (
                'CHAMPION_KILL', 'BUILDING_KILL', 'ELITE_MONSTER_KILL',
                'CHAMPION_SPECIAL_KILL', 'ITEM_PURCHASED',
                'SKILL_LEVEL_UP', 'WARD_PLACED', 'WARD_KILL'
            );
        CREATE INDEX IF NOT EXISTS idx_events_objective_match_ts
            ON events(match_id, timestamp_ms)
            WHERE event_type IN (
                'ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED'
            );
        CREATE INDEX IF NOT EXISTS idx_frames_puuid ON frames(puuid, timestamp_ms);

        CREATE TABLE IF NOT EXISTS player_match_stats (
            puuid TEXT,
            match_id TEXT,
            team_id INTEGER,
            role TEXT,
            created_at INTEGER,
            winning_team INTEGER,
            window_10_rows INTEGER DEFAULT 0,
            kda_10_sum REAL DEFAULT 0.0,
            cs_10_sum REAL DEFAULT 0.0,
            total_gold_10_sum REAL DEFAULT 0.0,
            ward_count_10_sum REAL DEFAULT 0.0,
            PRIMARY KEY (puuid, match_id)
        );
        CREATE INDEX IF NOT EXISTS idx_player_match_stats_match
            ON player_match_stats(match_id, puuid);
        CREATE INDEX IF NOT EXISTS idx_player_match_stats_puuid_created
            ON player_match_stats(puuid, created_at, match_id);

        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            name TEXT,
            explanation TEXT,
            feature_key TEXT,
            importance REAL,
            game_phase TEXT,
            category TEXT
        );

        CREATE TABLE IF NOT EXISTS key_moments (
            match_id TEXT,
            timestamp_ms INTEGER,
            concept_id TEXT,
            team_id INTEGER,
            description TEXT,
            impact_score REAL,
            PRIMARY KEY (match_id, timestamp_ms, concept_id)
        );

        CREATE TABLE IF NOT EXISTS player_progress (
            puuid TEXT,
            concept_id TEXT,
            times_seen INTEGER DEFAULT 0,
            times_correct INTEGER DEFAULT 0,
            last_seen INTEGER,
            PRIMARY KEY (puuid, concept_id)
        );
    """)
    conn.commit()
    conn.close()


def insert_game(conn, match_id, patch, queue_id, duration_s, winning_team, created_at):
    conn.execute(
        "INSERT OR IGNORE INTO games VALUES (?, ?, ?, ?, ?, ?)",
        (match_id, patch, queue_id, duration_s, winning_team, created_at)
    )


def insert_player(conn, puuid, riot_id, rank_tier=None, rank_division=None, lp=None):
    conn.execute(
        """INSERT INTO players (puuid, riot_id, rank_tier, rank_division, lp, last_updated)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(puuid) DO UPDATE SET
             riot_id=excluded.riot_id,
             rank_tier=COALESCE(excluded.rank_tier, rank_tier),
             rank_division=COALESCE(excluded.rank_division, rank_division),
             lp=COALESCE(excluded.lp, lp),
             last_updated=excluded.last_updated""",
        (puuid, riot_id, rank_tier, rank_division, lp, int(time.time()))
    )


def insert_frame(conn, match_id, slot, puuid, team_id, role, ts_ms,
                  current_gold, total_gold, xp, level, cs, jungle_cs,
                  pos_x, pos_y, kills, deaths, assists,
                  ward_count, item_ids):
    conn.execute(
        "INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (match_id, slot, puuid, team_id, role, ts_ms,
         current_gold, total_gold, xp, level, cs, jungle_cs,
         pos_x, pos_y, kills, deaths, assists,
         ward_count, json.dumps(item_ids))
    )


def insert_event(conn, match_id, timestamp_ms, event_type,
                  participant_id=None, killer_id=None, victim_id=None,
                  killer_team=None, team_id=None,
                  position_x=None, position_y=None, details='{}'):
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (match_id, timestamp_ms, event_type,
         participant_id, killer_id, victim_id,
         killer_team, team_id,
         position_x, position_y, details)
    )


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {os.path.abspath(DEFAULT_DB_PATH)}")
