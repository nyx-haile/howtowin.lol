#!/usr/bin/env python3
"""Merge a staging DB (from grow_games --db-dir) into the main DB.

Usage:
  python merge_db.py /path/to/staging/dir
  python merge_db.py /path/to/staging/dir --dry-run

Merges both howtowin.db and raw_matches.db. Existing rows are skipped
(INSERT OR IGNORE / ON CONFLICT DO NOTHING) so it's safe to re-run.
"""
import argparse
import os
import sqlite3
import sys
import time


def _attach_and_merge_main(main_path, staging_path, dry_run=False):
    """Merge staging howtowin.db into main howtowin.db."""
    if not os.path.exists(staging_path):
        print(f"  skip: {staging_path} does not exist")
        return 0

    conn = sqlite3.connect(main_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"ATTACH DATABASE ? AS staging", (staging_path,))

    # Count what we're merging.
    new_games = conn.execute(
        "SELECT COUNT(*) FROM staging.games WHERE match_id NOT IN (SELECT match_id FROM main.games)"
    ).fetchone()[0]
    new_players = conn.execute(
        "SELECT COUNT(*) FROM staging.players WHERE puuid NOT IN (SELECT puuid FROM main.players)"
    ).fetchone()[0]
    new_frames = conn.execute(
        "SELECT COUNT(*) FROM staging.frames f "
        "WHERE NOT EXISTS (SELECT 1 FROM main.frames m "
        "WHERE m.match_id=f.match_id AND m.participant_slot=f.participant_slot "
        "AND m.timestamp_ms=f.timestamp_ms)"
    ).fetchone()[0]

    print(f"  games: +{new_games}  players: +{new_players}  frames: +{new_frames}")

    if dry_run:
        conn.close()
        return new_games

    conn.execute("INSERT OR IGNORE INTO main.games SELECT * FROM staging.games")
    conn.execute(
        """INSERT INTO main.players SELECT * FROM staging.players
           WHERE TRUE ON CONFLICT(puuid) DO UPDATE SET
             riot_id=COALESCE(excluded.riot_id, riot_id),
             rank_tier=COALESCE(excluded.rank_tier, rank_tier),
             rank_division=COALESCE(excluded.rank_division, rank_division),
             lp=COALESCE(excluded.lp, lp),
             last_updated=MAX(excluded.last_updated, last_updated)"""
    )
    conn.execute("INSERT OR IGNORE INTO main.frames SELECT * FROM staging.frames")
    conn.execute("INSERT OR IGNORE INTO main.events SELECT * FROM staging.events")
    conn.commit()
    conn.close()
    return new_games


def _attach_and_merge_raw(main_path, staging_path, dry_run=False):
    """Merge staging raw_matches.db into main raw_matches.db."""
    if not os.path.exists(staging_path):
        print(f"  skip: {staging_path} does not exist")
        return 0

    conn = sqlite3.connect(main_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"ATTACH DATABASE ? AS staging", (staging_path,))

    new_raw = conn.execute(
        "SELECT COUNT(*) FROM staging.raw_matches WHERE match_id NOT IN (SELECT match_id FROM main.raw_matches)"
    ).fetchone()[0]
    print(f"  raw_matches: +{new_raw}")

    if dry_run:
        conn.close()
        return new_raw

    conn.execute("INSERT OR IGNORE INTO main.raw_matches SELECT * FROM staging.raw_matches")
    conn.commit()
    conn.close()
    return new_raw


def main():
    parser = argparse.ArgumentParser(description="Merge staging DBs into main DBs.")
    parser.add_argument("staging_dir", help="Directory containing staging howtowin.db and raw_matches.db")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged without writing.")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    main_db = os.path.join(data_dir, 'howtowin.db')
    main_raw = os.path.join(data_dir, 'raw_matches.db')
    staging_db = os.path.join(args.staging_dir, 'howtowin.db')
    staging_raw = os.path.join(args.staging_dir, 'raw_matches.db')

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"{mode}Merging {args.staging_dir} -> {data_dir}")

    print("howtowin.db:")
    new_games = _attach_and_merge_main(main_db, staging_db, args.dry_run)

    print("raw_matches.db:")
    _attach_and_merge_raw(main_raw, staging_raw, args.dry_run)

    if not args.dry_run and new_games > 0:
        print(f"\nMerged {new_games} new games. You may want to re-run holdout splits.")


if __name__ == "__main__":
    main()
