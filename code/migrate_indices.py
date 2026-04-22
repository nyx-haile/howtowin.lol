#!/usr/bin/env python3
"""Apply missing indices to an existing howtowin.db.

Safe to re-run: every statement is `CREATE INDEX IF NOT EXISTS`. Run
against any data/howtowin.db after a merge or when adding indices in
code/db.py to bring an older DB forward.

Usage:
  uv run python migrate_indices.py                 # default DEFAULT_DB_PATH
  uv run python migrate_indices.py /path/to/db     # explicit path
"""
import sys
import time

from db import get_conn

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_type  ON events(match_id, event_type)",
    "CREATE INDEX IF NOT EXISTS idx_events_match_ts ON events(match_id, timestamp_ms)",
    """CREATE INDEX IF NOT EXISTS idx_events_meaningful_match_ts
       ON events(match_id, timestamp_ms)
       WHERE event_type IN (
           'CHAMPION_KILL', 'BUILDING_KILL', 'ELITE_MONSTER_KILL',
           'CHAMPION_SPECIAL_KILL', 'ITEM_PURCHASED',
           'SKILL_LEVEL_UP', 'WARD_PLACED', 'WARD_KILL'
       )""",
    """CREATE INDEX IF NOT EXISTS idx_events_objective_match_ts
       ON events(match_id, timestamp_ms)
       WHERE event_type IN (
           'ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED'
       )""",
    "CREATE INDEX IF NOT EXISTS idx_frames_puuid ON frames(puuid, timestamp_ms)",
]


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    conn = get_conn(db_path)
    for stmt in INDICES:
        name = stmt.split()[5]
        t0 = time.time()
        conn.execute(stmt)
        print(f"  {name:24s}  {time.time() - t0:6.2f}s")
    conn.commit()
    conn.close()
    print("done")


if __name__ == "__main__":
    main()
