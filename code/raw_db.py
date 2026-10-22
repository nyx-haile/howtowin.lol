import sqlite3
import json
import gzip
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from runtime_paths import resolve_runtime_paths

DEFAULT_RAW_DB_PATH = str(resolve_runtime_paths().raw_db_path)


def _resolve_raw_db_path(db_path=None):
    if db_path:
        return db_path
    return str(resolve_runtime_paths().raw_db_path)


def _is_sqlite_uri(db_path):
    return isinstance(db_path, str) and db_path.startswith("file:")


def _uri_requests_readonly(db_path):
    if not _is_sqlite_uri(db_path):
        return False
    query = parse_qs(urlsplit(db_path).query)
    modes = [value.lower() for value in query.get("mode", [])]
    immutable = [value.lower() for value in query.get("immutable", [])]
    return "ro" in modes or any(value in {"1", "true", "yes", "on"} for value in immutable)


def _readonly_uri(db_path):
    if not _is_sqlite_uri(db_path):
        return f"{Path(db_path).expanduser().resolve().as_uri()}?mode=ro"
    parts = urlsplit(db_path)
    query = parse_qs(parts.query, keep_blank_values=True)
    if "mode" not in query:
        query["mode"] = ["ro"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))


def _connect_args(db_path, readonly):
    resolved = str(db_path)
    use_readonly = bool(readonly) or _uri_requests_readonly(resolved)
    if use_readonly:
        return _readonly_uri(resolved), True, True
    return resolved, _is_sqlite_uri(resolved), False


def _apply_read_pragmas(conn):
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=8589934592")
    conn.execute("PRAGMA foreign_keys=ON")


def get_raw_conn(db_path=None, readonly=None):
    target, uri, use_readonly = _connect_args(_resolve_raw_db_path(db_path), readonly)
    conn = sqlite3.connect(target, timeout=30.0, uri=uri)
    if use_readonly:
        conn.execute("PRAGMA query_only=ON")
        _apply_read_pragmas(conn)
        return conn
    conn.execute("PRAGMA journal_mode=WAL")
    _apply_read_pragmas(conn)
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


def get_raw_match(match_id, db_path=None, include_timeline=True, readonly=None):
    conn = get_raw_conn(db_path, readonly=readonly)
    if include_timeline:
        row = conn.execute(
            "SELECT match_json, timeline_json FROM raw_matches WHERE match_id = ?",
            (match_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT match_json FROM raw_matches WHERE match_id = ?",
            (match_id,)
        ).fetchone()
    conn.close()
    if row is None:
        return None, None
    match = json.loads(gzip.decompress(row[0]).decode())
    timeline = json.loads(gzip.decompress(row[1]).decode()) if include_timeline else None
    return match, timeline


def get_raw_match_only(match_id, db_path=None, readonly=None):
    match, _timeline = get_raw_match(match_id, db_path=db_path, include_timeline=False, readonly=readonly)
    return match


if __name__ == "__main__":
    init_raw_db()
    print(f"Raw matches DB initialized at {os.path.abspath(DEFAULT_RAW_DB_PATH)}")
