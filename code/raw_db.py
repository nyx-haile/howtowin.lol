import sqlite3
import json
import gzip
import os
import time

DEFAULT_RAW_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_matches.db')


def _resolve_raw_db_path(db_path=None):
    if db_path:
        return db_path
    return os.environ.get('HOWL_RAW_DB_PATH', DEFAULT_RAW_DB_PATH)


def get_raw_conn(db_path=None):
    conn = sqlite3.connect(_resolve_raw_db_path(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_raw_db(db_path=None):
    conn = get_raw_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS raw_matches (
            match_id TEXT PRIMARY KEY,
            match_json BLOB,
            timeline_json BLOB,
            stored_at INTEGER
        );
    """)
    conn.commit()
    conn.close()


def insert_raw_match(conn, match_id, match_dict, timeline_dict):
    match_gz = gzip.compress(json.dumps(match_dict, separators=(',', ':')).encode())
    timeline_gz = gzip.compress(json.dumps(timeline_dict, separators=(',', ':')).encode())
    conn.execute(
        "INSERT OR REPLACE INTO raw_matches VALUES (?, ?, ?, ?)",
        (match_id, match_gz, timeline_gz, int(time.time()))
    )


def get_raw_match(match_id, db_path=None):
    conn = get_raw_conn(db_path)
    row = conn.execute(
        "SELECT match_json, timeline_json FROM raw_matches WHERE match_id = ?",
        (match_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None, None
    match = json.loads(gzip.decompress(row[0]).decode())
    timeline = json.loads(gzip.decompress(row[1]).decode())
    return match, timeline


if __name__ == "__main__":
    init_raw_db()
    print(f"Raw matches DB initialized at {os.path.abspath(DEFAULT_RAW_DB_PATH)}")
