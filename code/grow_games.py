#!/usr/bin/env python3
"""Grow the local games DB to a target match count with a live status bar."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from db import get_conn, init_db
from redis_init import clear_queues, init_rate_limits
from seed import seed_top_players


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


def run_queue_once(code_dir: str, target: int, start_count: int, start_ts: float, poll_s: float) -> None:
    proc = subprocess.Popen(
        [sys.executable, os.path.join("test", "queue_run.py")],
        cwd=code_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    while proc.poll() is None:
        render_progress(get_game_count(), target, start_count, start_ts)
        time.sleep(poll_s)

    if proc.returncode != 0:
        raise RuntimeError(f"queue_run.py exited with code {proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grow the games DB to a target count.")
    parser.add_argument("--target", type=int, default=10_000, help="Desired total games in DB (default: 10000).")
    parser.add_argument("--seed-top", action="store_true", help="Seed challenger+grandmaster players before crawling.")
    parser.add_argument("--per-tier", type=int, default=50, help="Players per tier when --seed-top is used.")
    parser.add_argument("--clear-queues", action="store_true", help="Clear redis queues/tracking keys before crawling.")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Progress bar refresh interval in seconds.")
    parser.add_argument(
        "--max-stagnant-runs",
        type=int,
        default=3,
        help="Fail after this many queue runs with no new games.",
    )
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.abspath(__file__))
    init_db()

    if args.clear_queues:
        clear_queues()
    init_rate_limits()
    if args.seed_top:
        seed_top_players(per_tier=args.per_tier)

    start_ts = time.time()
    start_count = get_game_count()
    current = start_count
    stagnant_runs = 0

    print(f"Starting from {start_count} games; target={args.target}")
    render_progress(current, args.target, start_count, start_ts)

    while current < args.target:
        run_queue_once(code_dir, args.target, start_count, start_ts, args.poll_seconds)
        new_count = get_game_count()
        render_progress(new_count, args.target, start_count, start_ts)

        if new_count <= current:
            stagnant_runs += 1
            if stagnant_runs >= args.max_stagnant_runs:
                print()
                raise RuntimeError(
                    f"No growth after {stagnant_runs} queue runs; stopped at {new_count}/{args.target}."
                )
        else:
            stagnant_runs = 0
        current = new_count

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
