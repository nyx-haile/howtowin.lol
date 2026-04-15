# code/tests/model/conftest.py
import os
import sys
import pytest

# Make `import model.xxx` work when running pytest from code/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FIXTURE_MATCH_ID = "NA1_5434372583"


@pytest.fixture(scope="session")
def fixture_match_id():
    return FIXTURE_MATCH_ID


@pytest.fixture(scope="session")
def all_match_ids():
    """All parsed match_ids in the corpus. Used to exercise dataset loading."""
    from db import get_conn
    conn = get_conn()
    rows = conn.execute("SELECT match_id FROM games ORDER BY created_at ASC").fetchall()
    conn.close()
    return [r["match_id"] for r in rows]
