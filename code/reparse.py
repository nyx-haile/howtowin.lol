#!/usr/bin/env python3
"""Re-parse raw_matches that aren't in the games table.

Useful after a DB merge or interrupted run where raw match data was fetched
but parsing into games/frames/events didn't complete.

Usage:
  uv run python reparse.py
  uv run python reparse.py --dry-run
"""
import argparse
import sys
import traceback

from db import get_conn, init_db
from raw_db import get_raw_conn, get_raw_match
from parser import parser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print what would be parsed without doing it.")
    args = ap.parse_args()

    init_db()

    conn = get_conn()
    raw_conn = get_raw_conn()

    raw_ids = {r[0] for r in raw_conn.execute("SELECT match_id FROM raw_matches").fetchall()}
    parsed_ids = {r[0] for r in conn.execute("SELECT match_id FROM games").fetchall()}
    conn.close()
    raw_conn.close()

    to_parse = sorted(raw_ids - parsed_ids)
    print(f"raw_matches: {len(raw_ids)}  games: {len(parsed_ids)}  to parse: {len(to_parse)}")

    if args.dry_run or not to_parse:
        return

    p = parser.connect()
    ok = 0
    fail = 0
    for i, match_id in enumerate(to_parse, 1):
        match_data, timeline_data = get_raw_match(match_id)
        if not match_data or not timeline_data:
            print(f"[{i}/{len(to_parse)}] SKIP {match_id} — missing data in raw_matches")
            fail += 1
            continue
        try:
            p.match = match_id
            p.handle_match(match_data=match_data, match_timeline=timeline_data)
            print(f"[{i}/{len(to_parse)}] OK {match_id}", flush=True)
            ok += 1
        except Exception:
            print(f"[{i}/{len(to_parse)}] FAIL {match_id}")
            traceback.print_exc()
            fail += 1

    print(f"\nDone. parsed={ok} failed={fail}")


if __name__ == "__main__":
    raise SystemExit(main())
