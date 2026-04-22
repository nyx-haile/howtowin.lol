import os
import sys
import tempfile

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
