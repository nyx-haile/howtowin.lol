"""Backfill players.rank_tier/division/lp for UNKNOWN puuids.

For each puuid with NULL rank_tier but at least one recorded match, we:
1. Infer platform from the first match_id prefix (e.g. "NA1_..." -> "na1")
2. Fetch summoner via summoner-v4 (by-puuid)
3. Fetch league entries via league-v4 (by-summoner)
4. Pick the RANKED_SOLO_5x5 entry and write tier/division/lp

Parallelised one thread per Riot routing region (americas, europe, asia, sea)
so each route gets its own rate-limit bucket.
"""
import argparse
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import agent, route_for_platform
from db import get_conn


ROUTES = ["americas", "europe", "asia", "sea"]


def _pick_solo_entry(entries):
    if not entries:
        return None
    solo = [e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"]
    if not solo:
        return None
    # In practice a puuid has at most one solo entry; pick highest LP to be safe.
    return max(solo, key=lambda e: int(e.get("leaguePoints", 0) or 0))


def _fetch_rank_for_puuid(a: "agent", puuid: str, platform: str):
    """Return (tier, division, lp) or (None, None, None) if no rank / API failure."""
    try:
        summoner = a.get_summoner_by_puuid(puuid, region=platform)
    except AssertionError:
        return None, None, None
    if not summoner or not summoner.get("id"):
        return None, None, None
    try:
        entries = a.get_league_entries_by_summoner(summoner["id"], region=platform)
    except AssertionError:
        return None, None, None
    solo = _pick_solo_entry(entries)
    if not solo:
        return None, None, None
    return solo.get("tier"), solo.get("rank"), int(solo.get("leaguePoints", 0) or 0)


def _collect_unknown_puuids():
    """Map route -> list of (puuid, platform). Skip puuids with no matches."""
    conn = get_conn()
    try:
        # Per-row LIMIT 1 subquery uses idx_frames_puuid directly; avoids
        # a full GROUP BY over the frames table (>1M rows under grow_games).
        rows = conn.execute(
            """
            SELECT p.puuid,
                   (SELECT f.match_id FROM frames f
                     WHERE f.puuid = p.puuid LIMIT 1) AS any_match
            FROM players p
            WHERE p.rank_tier IS NULL
            """
        ).fetchall()
        rows = [r for r in rows if r["any_match"]]
    finally:
        conn.close()

    per_route: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        match_id = r["any_match"]
        platform = match_id.split("_", 1)[0].lower() if "_" in match_id else "na1"
        route = route_for_platform(platform)
        per_route[route].append((r["puuid"], platform))
    return per_route


def _worker(route: str, work_q: "queue.Queue", counters: dict, lock: threading.Lock):
    """Drain work_q; each item is (puuid, platform). Writes to db via a dedicated
    connection per worker (safe: WAL + busy_timeout)."""
    a = agent.connect()
    conn = get_conn()
    processed = 0
    found = 0
    t0 = time.time()
    while True:
        try:
            item = work_q.get_nowait()
        except queue.Empty:
            break
        puuid, platform = item
        try:
            tier, div, lp = _fetch_rank_for_puuid(a, puuid, platform)
        except Exception:
            traceback.print_exc()
            work_q.task_done()
            continue

        if tier is not None:
            conn.execute(
                """
                UPDATE players
                   SET rank_tier = ?, rank_division = ?, lp = ?, last_updated = ?
                 WHERE puuid = ?
                """,
                (tier, div, lp, int(time.time()), puuid),
            )
            if processed % 50 == 0:
                conn.commit()
            found += 1
        processed += 1
        if processed % 200 == 0:
            with lock:
                counters[route] = (processed, found)
            rate = processed / max(time.time() - t0, 1e-3)
            print(f"[{route}] processed={processed} found={found} rate={rate:.1f}/s",
                  flush=True)
        work_q.task_done()
    conn.commit()
    conn.close()
    with lock:
        counters[route] = (processed, found)
    print(f"[{route}] DONE processed={processed} found={found} "
          f"elapsed={int(time.time() - t0)}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads-per-route", type=int, default=2,
                    help="Worker threads per route (default: 2)")
    ap.add_argument("--limit-per-route", type=int, default=0,
                    help="Cap items per route (0 = no limit)")
    args = ap.parse_args()

    print("Collecting UNKNOWN puuids...", flush=True)
    per_route = _collect_unknown_puuids()
    total = sum(len(v) for v in per_route.values())
    print(f"Found {total} puuids to backfill:", flush=True)
    for r in ROUTES:
        n = len(per_route.get(r, []))
        print(f"  {r}: {n}", flush=True)

    counters: dict[str, tuple[int, int]] = {}
    lock = threading.Lock()

    threads: list[threading.Thread] = []
    for route in ROUTES:
        items = per_route.get(route, [])
        if args.limit_per_route > 0:
            items = items[: args.limit_per_route]
        if not items:
            continue
        work_q: queue.Queue = queue.Queue()
        for it in items:
            work_q.put(it)
        for i in range(args.threads_per_route):
            t = threading.Thread(
                target=_worker, args=(route, work_q, counters, lock),
                name=f"rank-{route}-{i}", daemon=True,
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    total_processed = sum(p for p, _ in counters.values())
    total_found = sum(f for _, f in counters.values())
    print(f"ALL DONE processed={total_processed} found={total_found} "
          f"({100 * total_found / max(total_processed, 1):.1f}% ranked)")


if __name__ == "__main__":
    main()
