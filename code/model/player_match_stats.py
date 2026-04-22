"""Materialized per-player per-match aggregates for training features."""
from __future__ import annotations

from db import get_conn

PLAYER_MATCH_STATS_TABLE = "player_match_stats"

DDL = f"""
CREATE TABLE IF NOT EXISTS {PLAYER_MATCH_STATS_TABLE} (
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
    ON {PLAYER_MATCH_STATS_TABLE}(match_id, puuid);
CREATE INDEX IF NOT EXISTS idx_player_match_stats_puuid_created
    ON {PLAYER_MATCH_STATS_TABLE}(puuid, created_at, match_id);
"""

_ready_db_paths: set[str] = set()


def _source_match_count(conn) -> int:
    return conn.execute(
        """
        SELECT COUNT(DISTINCT match_id)
        FROM frames
        WHERE participant_slot BETWEEN 1 AND 10
        """
    ).fetchone()[0]


def _covered_match_count(conn) -> int:
    return conn.execute(
        f"SELECT COUNT(DISTINCT match_id) FROM {PLAYER_MATCH_STATS_TABLE}"
    ).fetchone()[0]


def _db_path_for_conn(conn) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None:
        return ""
    return row[2] or ""


def ensure_player_match_stats(conn=None):
    own_conn = conn is None
    if own_conn:
        conn = get_conn()

    db_path = _db_path_for_conn(conn)
    if db_path in _ready_db_paths:
        if own_conn:
            conn.close()
        return

    conn.executescript(DDL)

    total_matches = _source_match_count(conn)
    covered_matches = _covered_match_count(conn)

    if covered_matches < total_matches:
        conn.execute(
            f"""
            WITH missing AS (
                SELECT DISTINCT match_id
                FROM frames
                WHERE participant_slot BETWEEN 1 AND 10
                EXCEPT
                SELECT DISTINCT match_id
                FROM {PLAYER_MATCH_STATS_TABLE}
            ),
            agg AS (
                SELECT
                    f.puuid AS puuid,
                    f.match_id AS match_id,
                    MIN(f.team_id) AS team_id,
                    MIN(f.role) AS role,
                    SUM(CASE WHEN f.timestamp_ms BETWEEN 540000 AND 660000 THEN 1 ELSE 0 END) AS window_10_rows,
                    SUM(CASE
                            WHEN f.timestamp_ms BETWEEN 540000 AND 660000
                            THEN ((f.kills + f.assists) * 1.0) / MAX(f.deaths, 1)
                            ELSE 0.0
                        END) AS kda_10_sum,
                    SUM(CASE WHEN f.timestamp_ms BETWEEN 540000 AND 660000 THEN f.cs ELSE 0.0 END) AS cs_10_sum,
                    SUM(CASE WHEN f.timestamp_ms BETWEEN 540000 AND 660000 THEN f.total_gold ELSE 0.0 END) AS total_gold_10_sum,
                    SUM(CASE WHEN f.timestamp_ms BETWEEN 540000 AND 660000 THEN f.ward_count ELSE 0.0 END) AS ward_count_10_sum
                FROM frames f
                JOIN missing m ON m.match_id = f.match_id
                WHERE f.participant_slot BETWEEN 1 AND 10
                GROUP BY f.puuid, f.match_id
            )
            INSERT OR REPLACE INTO {PLAYER_MATCH_STATS_TABLE} (
                puuid, match_id, team_id, role, created_at, winning_team,
                window_10_rows, kda_10_sum, cs_10_sum, total_gold_10_sum, ward_count_10_sum
            )
            SELECT
                a.puuid,
                a.match_id,
                a.team_id,
                a.role,
                g.created_at,
                g.winning_team,
                a.window_10_rows,
                a.kda_10_sum,
                a.cs_10_sum,
                a.total_gold_10_sum,
                a.ward_count_10_sum
            FROM agg a
            JOIN games g USING (match_id)
            """
        )
        conn.commit()

    _ready_db_paths.add(db_path)
    if own_conn:
        conn.close()
