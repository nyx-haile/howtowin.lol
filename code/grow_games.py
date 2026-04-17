#!/usr/bin/env python3
"""Grow the local games DB to a target match count with a live status bar."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

from db import get_conn, init_db
from fetch import agent
from redis_init import clear_queues, init_rate_limits
from seed import bump_players_by_rank, seed_riot_ids, seed_top_players


def recover_in_flight():
    """Move items stuck in processing sets back to their queues.

    When a run is interrupted, players/matches mid-processing never finish.
    This pushes them back into the work queues so the next run picks them up.
    """
    r = agent.connect()

    stale_players = r.smembers("player_processing")
    if stale_players:
        for puuid in stale_players:
            r.zincrby("player_queue", 1, puuid)
        r.delete("player_processing")
        print(f"Recovered {len(stale_players)} players from stale processing state")

    stale_matches = r.smembers("match_processing")
    if stale_matches:
        for mid in stale_matches:
            r.zincrby("match_queue", 1, mid)
        r.delete("match_processing")
        print(f"Recovered {len(stale_matches)} matches from stale processing state")


def get_game_count() -> int:
    conn = get_conn()
    try:
        return int(conn.execute("SELECT COUNT(*) FROM games").fetchone()[0])
    finally:
        conn.close()


def render_progress(current: int, target: int, start_count: int, start_ts: float, width: int = 40) -> None:
    target = max(target, 1)
    ratio = min(max(current / target, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)

    elapsed = max(time.time() - start_ts, 1e-6)
    gained = max(current - start_count, 0)
    rate = gained / elapsed
    remaining = max(target - current, 0)
    eta_s = int(remaining / rate) if rate > 0 else -1
    eta = f"{eta_s}s" if eta_s >= 0 else "?"

    line = (
        f"\r[{bar}] {current}/{target} "
        f"({ratio * 100:5.1f}%) +{gained} @ {rate:.2f} games/s ETA {eta}"
    )
    print(line, end="", flush=True)


def run_queue_once(code_dir: str, target: int, start_count: int, start_ts: float,
                   poll_s: float, n_workers: int = 1) -> None:
    log_dir = os.path.join(code_dir, '..', 'data', 'worker_logs')
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    log_files = []
    for i in range(n_workers):
        lf = open(os.path.join(log_dir, f'worker_{i}.log'), 'w')
        log_files.append(lf)
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join("test", "queue_run.py")],
            cwd=code_dir,
            stdout=lf,
            stderr=subprocess.STDOUT,
        ))

    while any(p.poll() is None for p in procs):
        render_progress(get_game_count(), target, start_count, start_ts)
        time.sleep(poll_s)

    for lf in log_files:
        lf.close()

    failed = [(i, p.returncode) for i, p in enumerate(procs) if p.returncode and p.returncode != 0]
    if failed:
        for i, code in failed:
            log_path = os.path.join(log_dir, f'worker_{i}.log')
            print(f"\n--- worker {i} (exit {code}) last 20 lines: {log_path} ---")
            try:
                lines = open(log_path).readlines()
                print(''.join(lines[-20:]))
            except Exception:
                pass
        raise RuntimeError(f"Workers failed: {failed}")


def main() -> int:
    bump_tier_plan = ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]

    parser = argparse.ArgumentParser(description="Grow the games DB to a target count.")
    parser.add_argument("--target", type=int, default=10_000, help="Desired total games in DB (default: 10000).")
    parser.add_argument("--seed-top", action="store_true", help="Seed challenger players across all servers before crawling.")
    parser.add_argument("--refresh-seed-cache", action="store_true", help="Force re-fetch seed list from Riot API even if cache is fresh.")
    parser.add_argument("--per-tier", type=int, default=50, help="Players per tier when --seed-top is used.")
    parser.add_argument(
        "--seed-regions",
        type=str,
        default=None,
        help="Comma-separated platform regions for top seeding (default: all supported regions).",
    )
    parser.add_argument(
        "--seed-player",
        action="append",
        default=[],
        help="Riot ID to seed (name#tag). Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-grandmaster",
        action="store_true",
        help="Seed challenger only (skip grandmaster).",
    )
    parser.add_argument(
        "--bump-limit",
        type=int,
        default=500,
        help="Maximum queued players to bump per stagnation recovery step.",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="Write to separate DBs in this directory (for parallel growth while training reads the main DB).",
    )
    parser.add_argument("--clear-queues", action="store_true", help="Clear redis queues/tracking keys before crawling.")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel queue_run.py workers (default: 3).")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Progress bar refresh interval in seconds.")
    parser.add_argument(
        "--max-stagnant-runs",
        type=int,
        default=3,
        help="Fail after this many queue runs with no new games.",
    )
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.abspath(__file__))

    if args.db_dir:
        os.makedirs(args.db_dir, exist_ok=True)
        data_dir = os.path.join(code_dir, '..', 'data')
        staging_db = os.path.join(args.db_dir, 'howtowin.db')
        staging_raw = os.path.join(args.db_dir, 'raw_matches.db')
        # Seed staging dir from main DBs so we don't re-fetch existing games.
        for src, dst in [(os.path.join(data_dir, 'howtowin.db'), staging_db),
                         (os.path.join(data_dir, 'raw_matches.db'), staging_raw)]:
            if not os.path.exists(dst) and os.path.exists(src):
                print(f"Copying {src} -> {dst}")
                shutil.copy2(src, dst)
        os.environ['HOWL_DB_PATH'] = staging_db
        os.environ['HOWL_RAW_DB_PATH'] = staging_raw
        print(f"Writing to separate DBs in {args.db_dir}")

    init_db()

    if args.clear_queues:
        clear_queues()
    init_rate_limits()
    recover_in_flight()

    riot_ids = ["chaos#oda", *args.seed_player]
    deduped_riot_ids = list(dict.fromkeys(riot_ids))
    if deduped_riot_ids:
        print(f"Seeding Riot IDs: {', '.join(deduped_riot_ids)}")
        seed_riot_ids(deduped_riot_ids)

    if args.seed_top:
        regions = None
        if args.seed_regions:
            regions = [r.strip() for r in args.seed_regions.split(",") if r.strip()]
        seed_top_players(
            per_tier=args.per_tier,
            regions=regions,
            include_grandmaster=not args.no_grandmaster,
            refresh_cache=args.refresh_seed_cache,
        )

    start_ts = time.time()
    start_count = get_game_count()
    current = start_count
    stagnant_runs = 0
    bump_idx = 0

    print(f"Starting from {start_count} games; target={args.target}")
    render_progress(current, args.target, start_count, start_ts)

    while current < args.target:
        run_queue_once(code_dir, args.target, start_count, start_ts, args.poll_seconds, args.workers)
        new_count = get_game_count()
        render_progress(new_count, args.target, start_count, start_ts)

        if new_count <= current:
            stagnant_runs += 1
            if stagnant_runs >= args.max_stagnant_runs:
                bump_tier = bump_tier_plan[min(bump_idx, len(bump_tier_plan) - 1)]
                bumped = bump_players_by_rank(min_tier=bump_tier, limit=args.bump_limit)
                if bump_idx < len(bump_tier_plan) - 1:
                    bump_idx += 1
                if bumped > 0:
                    stagnant_runs = 0
                    print(
                        f"\nNo growth detected; bumped {bumped} queued players at {bump_tier}+ priority and retrying."
                    )
                    continue
                print()
                raise RuntimeError(
                    f"No growth after {stagnant_runs} queue runs; no {bump_tier}+ players available to bump."
                )
        else:
            stagnant_runs = 0
        current = new_count

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
