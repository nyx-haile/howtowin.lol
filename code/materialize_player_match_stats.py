#!/usr/bin/env python3
"""Build or refresh the player_match_stats summary table for an existing DB."""
import sys
import time

from db import get_conn
from model.player_match_stats import PLAYER_MATCH_STATS_TABLE, ensure_player_match_stats


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    conn = get_conn(db_path)
    t0 = time.time()
    ensure_player_match_stats(conn)
    n_rows = conn.execute(f"SELECT COUNT(*) FROM {PLAYER_MATCH_STATS_TABLE}").fetchone()[0]
    n_matches = conn.execute(f"SELECT COUNT(DISTINCT match_id) FROM {PLAYER_MATCH_STATS_TABLE}").fetchone()[0]
    conn.close()
    print(f"{PLAYER_MATCH_STATS_TABLE}: {n_rows} rows across {n_matches} matches in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
