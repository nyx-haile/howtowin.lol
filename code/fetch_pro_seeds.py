#!/usr/bin/env python3
"""Fetch pro player Riot IDs from trackingthepros.com.

Reads the player list API (DataTables JSON endpoint) and scrapes each
player page for their account names, then writes a JSON seed file and
optionally seeds directly into the per-route player queues.

Usage examples:
    # All Diamond+ KR players → data/pro_seeds.json
    uv run python code/fetch_pro_seeds.py --route asia

    # Master+ only, seed directly into Redis
    uv run python code/fetch_pro_seeds.py --route asia --min-rank master --seed

    # All regions, Diamond+, output to custom path
    uv run python code/fetch_pro_seeds.py --out data/all_pro_seeds.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import requests

BASE = "https://www.trackingthepros.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; howtowin-seed/1.0)"}

# TrackingThePros region codes → Riot routing region
REGION_TO_ROUTE: dict[str, str] = {
    "NA": "americas", "LCS": "americas", "LLA": "americas",
    "CBLOL": "americas", "LCX": "americas",
    "EU": "europe", "LEC": "europe", "MENA": "europe",
    "TR": "europe", "CIS": "europe",
    "KR": "asia", "LCK": "asia", "JP": "asia", "LJL": "asia",
    "CN": "asia", "LPL": "asia",
    "SEA": "sea", "PCS": "sea", "VCS": "sea", "LCO": "sea",
}

# rankHighNum encodes tier*10000 + lp (Challenger=27, GM=26, Master=25, Diamond=24, …)
TIER_NUM: dict[str, int] = {
    "challenger": 27, "grandmaster": 26, "master": 25,
    "diamond": 24, "emerald": 23, "platinum": 22,
    "gold": 21, "silver": 20, "bronze": 19, "iron": 18,
}


def fetch_all_players(session: requests.Session, min_rank_num: int) -> list[dict]:
    page_size = 500
    players: list[dict] = []
    total = 999999
    start = 0

    def params(s: int) -> dict:
        return {
            "draw": "1", "start": str(s), "length": str(page_size),
            "search[value]": "", "search[regex]": "false",
            "columns[0][data]": "player_name", "columns[0][name]": "players.name",
            "columns[1][data]": "team_name",   "columns[1][name]": "teams.name",
            "columns[2][data]": "current_region", "columns[2][name]": "players.current_region",
            "columns[3][data][_]": "rankHigh", "columns[3][data][sort]": "rankHighNum",
            "columns[3][name]": "players.highest_rank",
            "columns[4][data]": "player_accounts", "columns[4][name]": "player_accounts",
            "order[0][column]": "3", "order[0][dir]": "desc",
        }

    while start < total:
        r = session.get(f"{BASE}/d/list_players", params=params(start), timeout=30)
        r.raise_for_status()
        data = r.json()
        total = data["recordsTotal"]
        batch = data["data"]
        players.extend(batch)
        start += len(batch)
        print(f"  fetched {start}/{total} players", end="\r", flush=True)

    print()
    return [p for p in players if (p.get("rankHighNum") or 0) >= min_rank_num]


_ACCOUNT_RE = re.compile(r'<td><b>\[(\w+)\]</b>\s*([^<]+?)\s*</td>')


def fetch_accounts(session: requests.Session, name_plug: str, delay: float) -> list[dict]:
    try:
        r = session.get(f"{BASE}/player/{name_plug}", timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  WARN {name_plug}: {e}")
        return []
    finally:
        time.sleep(delay)

    accounts = []
    for m in _ACCOUNT_RE.finditer(r.text):
        riot_id = m.group(2).strip()
        if "#" in riot_id:
            accounts.append({"region": m.group(1), "riot_id": riot_id})
    return accounts


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch pro Riot IDs from trackingthepros.com")
    ap.add_argument("--min-rank", default="diamond", choices=list(TIER_NUM),
                    help="Minimum rank tier (default: diamond)")
    ap.add_argument("--region", default=None,
                    help="Filter by TTP region code, e.g. KR, EU, NA (default: all)")
    ap.add_argument("--route", default=None,
                    choices=["americas", "europe", "asia", "sea"],
                    help="Filter by Riot routing region instead of exact code")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: data/pro_seeds_{route|all}.json)")
    ap.add_argument("--seed", action="store_true",
                    help="Seed matching accounts directly into per-route Redis queues")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="Seconds between player page fetches (default: 0.3)")
    args = ap.parse_args()

    min_rank_num = TIER_NUM[args.min_rank] * 10000

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"Fetching player list (min rank: {args.min_rank}, ≥{min_rank_num})...")
    players = fetch_all_players(session, min_rank_num)
    print(f"  {len(players)} players pass rank filter")

    if args.region:
        players = [p for p in players
                   if (p.get("current_region") or "").upper() == args.region.upper()]
        print(f"  {len(players)} players in region {args.region}")
    elif args.route:
        players = [p for p in players
                   if REGION_TO_ROUTE.get((p.get("current_region") or "").upper()) == args.route]
        print(f"  {len(players)} players on route {args.route}")

    if not players:
        print("No players matched filters.")
        return 1

    print(f"Fetching accounts for {len(players)} players (delay={args.delay}s)...")
    results: list[dict] = []
    for i, p in enumerate(players, 1):
        label = f"[{i}/{len(players)}] {p['name']} ({p.get('current_region','?')}) {p.get('rankHigh','')}"
        print(f"  {label}", flush=True)
        accounts = fetch_accounts(session, p["name_plug"], args.delay)
        route = REGION_TO_ROUTE.get((p.get("current_region") or "").upper(), "americas")
        results.append({
            "name": p["name"],
            "region": p.get("current_region"),
            "route": route,
            "rank": p.get("rankHigh"),
            "accounts": accounts,
        })

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"pro_seeds_{args.route or args.region or 'all'}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} players → {out_path}")

    # Summary by route
    by_route: dict[str, int] = {}
    for r in results:
        by_route[r["route"]] = by_route.get(r["route"], 0) + len(r["accounts"])
    for route, n in sorted(by_route.items()):
        print(f"  {route}: {n} accounts")

    if args.seed:
        # Import seed helpers — script must be run from code/ or project root
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from seed import seed_riot_ids
        from fetch import route_for_platform, agent
        from redis_init import init_rate_limits

        init_rate_limits()
        r_conn = agent.connect()

        seeded = 0
        for player in results:
            for acc in player["accounts"]:
                rid = acc["riot_id"]
                acc_route = REGION_TO_ROUTE.get(acc["region"].upper(), player["route"])
                if "#" not in rid:
                    continue
                name, tag = rid.split("#", 1)
                account_data = r_conn.get_account_by_riot_id(name, tag, route=acc_route)
                if not account_data or not account_data.get("puuid"):
                    print(f"  SKIP (no puuid): {rid}")
                    continue
                puuid = account_data["puuid"]
                if not r_conn.sismember("player_handled", puuid):
                    r_conn.zadd(f"player_queue:{acc_route}", {puuid: 0})
                    r_conn.hset("player_region", puuid, acc_route)
                    seeded += 1

        print(f"\nSeeded {seeded} accounts into per-route player queues.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
