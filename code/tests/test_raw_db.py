import os
import sys
import tempfile
import sqlite3
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import raw_db


def test_get_raw_match_only_skips_timeline_decode():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'raw.db')
        raw_db.init_raw_db(test_path)
        conn = raw_db.get_raw_conn(test_path)
        match = {
            "metadata": {"matchId": "NA1_123"},
            "info": {"participants": [{"puuid": "p1"}], "queueId": 420},
        }
        timeline = {"info": {"frames": [1, 2, 3]}}
        raw_db.insert_raw_match(conn, "NA1_123", match, timeline)
        conn.commit()
        conn.close()

        match_only = raw_db.get_raw_match_only("NA1_123", db_path=test_path)
        full_match, full_timeline = raw_db.get_raw_match("NA1_123", db_path=test_path)
        no_timeline_match, no_timeline = raw_db.get_raw_match(
            "NA1_123", db_path=test_path, include_timeline=False
        )

        assert match_only == match
        assert full_match == match
        assert full_timeline == timeline
        assert no_timeline_match == match
        assert no_timeline is None


def test_readonly_uri_reads_raw_db_and_rejects_writes_without_wal_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, 'raw_readonly.db')
        raw_db.init_raw_db(test_path)
        conn = raw_db.get_raw_conn(test_path)
        match = {"metadata": {"matchId": "NA1_123"}, "info": {"queueId": 420}}
        timeline = {"info": {"frames": [{"timestamp": 0}]}}
        raw_db.insert_raw_match(conn, "NA1_123", match, timeline)
        conn.commit()
        conn.close()

        wal_path = test_path + '-wal'
        if os.path.exists(wal_path):
            os.remove(wal_path)

        readonly_uri = f"{Path(test_path).as_uri()}?mode=ro&immutable=1"
        conn = raw_db.get_raw_conn(readonly_uri)
        try:
            row = conn.execute("SELECT match_id FROM raw_matches WHERE match_id = ?", ("NA1_123",)).fetchone()
            assert row[0] == "NA1_123"
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            try:
                raw_db.insert_raw_match(conn, "NA1_blocked", match, timeline)
                conn.commit()
            except sqlite3.DatabaseError:
                pass
            else:
                raise AssertionError("read-only raw connection allowed a write")
        finally:
            conn.close()

        full_match, full_timeline = raw_db.get_raw_match("NA1_123", db_path=readonly_uri)
        assert full_match == match
        assert full_timeline == timeline
        assert not os.path.exists(wal_path)
