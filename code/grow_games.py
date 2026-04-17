#!/usr/bin/env python3
"""Grow the local games DB to a target match count with a live status bar."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import deque

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


BUCKET_S = 30
CHART_HEIGHT = 8
YLABEL_W = 6   # "  42 │" — 4 digits + space + bar
DONE_CHAR = '█'
LIVE_CHAR = '▒'


class ProgressTracker:
    def __init__(self, start_count: int, target: int, start_ts: float) -> None:
        self.start_count = start_count
        self.target = target
        self.start_ts = start_ts
        self._bucket_ts = start_ts
        self._bucket_count = start_count
        self._buckets: deque[int] = deque()
        self._bucket_times: deque[float] = deque()  # wall time each bucket started
        self._drawn = False

    def tick(self, current: int) -> None:
        now = time.time()
        if now - self._bucket_ts >= BUCKET_S:
            self._buckets.append(current - self._bucket_count)
            self._bucket_times.append(self._bucket_ts)
            self._bucket_ts = now
            self._bucket_count = current
        self._draw(current)

    def _draw(self, current: int) -> None:
        term_w = max(shutil.get_terminal_size((80, 24)).columns, 30)
        chart_cols = term_w - YLABEL_W
        live = current - self._bucket_count

        recent_vals = list(self._buckets)[-(chart_cols - 1):]
        recent_times = list(self._bucket_times)[-(chart_cols - 1):]
        pad = chart_cols - 1 - len(recent_vals)
        vals  = [0] * pad + recent_vals  + [live]
        times = [None] * pad + recent_times + [self._bucket_ts]

        # Scale Y to the visible data range so bars fill the chart height.
        active = [v for v in vals if v > 0]
        if active:
            min_val = min(active)
            max_val = max(active)
        else:
            min_val, max_val = 0, 1
        range_val = max(max_val - min_val, 1)

        def bar_h(v: int) -> int:
            if v == 0:
                return 0
            return 1 + round((v - min_val) / range_val * (CHART_HEIGHT - 1))

        mid_val = (min_val + max_val) // 2

        def ylabel(row: int) -> str:
            if row == 0:
                return f'{max_val:>4} │'
            if row == CHART_HEIGHT // 2:
                return f'{mid_val:>4} │'
            return '     │'

        lines = []
        for row in range(CHART_HEIGHT):
            row_chars = []
            for i, v in enumerate(vals):
                filled = row >= CHART_HEIGHT - bar_h(v)
                row_chars.append((LIVE_CHAR if i == chart_cols - 1 else DONE_CHAR) if filled else ' ')
            lines.append(ylabel(row) + ''.join(row_chars))

        sep = f'{min_val:>4} └' + '─' * chart_cols

        # Time axis: label every ~label_interval cols, anchored from right
        label_interval = max(10, chart_cols // 6)
        time_row = [' '] * term_w
        for i in range(chart_cols - 2, -1, -label_interval):
            t = times[i]
            if t is None:
                continue
            label = time.strftime('%H:%M', time.localtime(t))
            pos = YLABEL_W + i
            for j, ch in enumerate(label):
                if pos + j < term_w:
                    time_row[pos + j] = ch
        time_line = ''.join(time_row)

        total_gained = current - self.start_count
        rate = total_gained / max(time.time() - self.start_ts, 1e-6)
        eta_s = int((self.target - current) / rate) if rate > 0 else -1
        eta = f'{eta_s}s' if eta_s >= 0 else '?'
        status = (f'      {time.strftime("%H:%M:%S")}  {current}/{self.target}'
                  f'  +{live} live  {rate:.2f}/s  ETA {eta}')

        n_lines = CHART_HEIGHT + 3  # chart + sep + time axis + status
        if self._drawn:
            sys.stdout.write(f'\033[{n_lines}A')
        for line in lines:
            print(line[:term_w])
        print(sep[:term_w])
        print(time_line[:term_w])
        print(status[:term_w].ljust(term_w), flush=True)
        self._drawn = True


def run_queue_once(code_dir: str, tracker: ProgressTracker,
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
        tracker.tick(get_game_count())
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
    tracker = ProgressTracker(start_count, args.target, start_ts)

    while current < args.target:
        run_queue_once(code_dir, tracker, args.poll_seconds, args.workers)
        new_count = get_game_count()
        tracker.tick(new_count)

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
