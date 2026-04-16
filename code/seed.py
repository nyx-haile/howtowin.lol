import json
import os
import requests
import sys
import time

from fetch import agent, route_for_platform
from db import get_conn, init_db, insert_player

SEED_CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'seed_cache.json')
SEED_CACHE_MAX_AGE_S = 7 * 24 * 3600  # refresh weekly

PLATFORM_REGIONS = [
    "br1",
    "eun1",
    "euw1",
    "jp1",
    "kr",
    "la1",
    "la2",
    "me1",
    "na1",
    "oc1",
    "ru",
    "sg2",
    "tr1",
    "tw2",
    "vn2",
]

TIER_ORDER = {
    "IRON": 1,
    "BRONZE": 2,
    "SILVER": 3,
    "GOLD": 4,
    "PLATINUM": 5,
    "EMERALD": 6,
    "DIAMOND": 7,
    "MASTER": 8,
    "GRANDMASTER": 9,
    "CHALLENGER": 10,
}


def _tier_priority(tier, lp=0):
    t = TIER_ORDER.get((tier or "").upper(), 0)
    return (t * 1000) + max(int(lp or 0), 0)


def _normalize_regions(regions):
    if regions is None:
        return PLATFORM_REGIONS
    return [r.strip().lower() for r in regions if r and r.strip()]


def _fetch_top_players_from_api(per_tier, regions, include_grandmaster):
    """Fetch challenger/grandmaster puuids from Riot API and return a list of dicts."""
    seed = agent.connect()
    tiers = [("CHALLENGER", seed.get_challenger_league)]
    if include_grandmaster:
        tiers.append(("GRANDMASTER", seed.get_grandmaster_league))

    entries_out = []
    for region in _normalize_regions(regions):
        for tier, fetch_fn in tiers:
            try:
                league = fetch_fn(region=region)
            except requests.exceptions.ConnectionError as e:
                print(f"Skipping {tier} for {region}: {e}")
                continue
            if not league:
                print(f"Failed to fetch {tier} league for {region}")
                continue
            entries = sorted(
                league.get("entries", []),
                key=lambda e: e.get("leaguePoints", 0),
                reverse=True,
            )[:per_tier]
            for entry in entries:
                if entry.get("puuid"):
                    entries_out.append({
                        "puuid": entry["puuid"],
                        "tier": tier,
                        "division": entry.get("rank", "I"),
                        "lp": int(entry.get("leaguePoints", 0) or 0),
                        "region": region,
                    })
            print(f"Fetched {len(entries)} from {tier} ({region})")
    return entries_out


def seed_top_players(per_tier=50, regions=None, include_grandmaster=True, refresh_cache=False):
    init_db()
    conn = get_conn()
    seed = agent.connect()

    # Use cache if fresh; fetch from API otherwise.
    cache_valid = (
        not refresh_cache
        and regions is None  # custom region lists always bypass cache
        and os.path.exists(SEED_CACHE_PATH)
        and (time.time() - os.path.getmtime(SEED_CACHE_PATH)) < SEED_CACHE_MAX_AGE_S
    )
    if cache_valid:
        with open(SEED_CACHE_PATH) as f:
            all_entries = json.load(f)
        print(f"Using seed cache ({len(all_entries)} players, {int((time.time() - os.path.getmtime(SEED_CACHE_PATH))/3600)}h old)")
    else:
        print("Fetching fresh seed list from Riot API...")
        all_entries = _fetch_top_players_from_api(per_tier, regions, include_grandmaster)
        if regions is None:
            os.makedirs(os.path.dirname(SEED_CACHE_PATH), exist_ok=True)
            with open(SEED_CACHE_PATH, 'w') as f:
                json.dump(all_entries, f)
            print(f"Cached {len(all_entries)} players to {SEED_CACHE_PATH}")

    count = 0
    for entry in all_entries:
        puuid = entry["puuid"]
        insert_player(conn, puuid, None, entry["tier"], entry["division"], entry["lp"])
        if not seed.sismember("player_handled", puuid):
            seed.zincrby("player_queue", _tier_priority(entry["tier"], entry["lp"]), puuid)
            seed.hset("player_region", puuid, route_for_platform(entry["region"]))
        count += 1

    conn.commit()
    conn.close()
    print(f"Seeded {count} players total")


def seed_single(game_name, tag_line):
    init_db()
    seed = agent.connect()
    account = seed.get_account_by_riot_id(game_name, tag_line)
    if account:
        seed.zadd("player_queue", {account["puuid"]: 0})
        conn = get_conn()
        insert_player(conn, account["puuid"], f"{game_name}#{tag_line}")
        conn.commit()
        conn.close()
        print(f"Seeded {game_name}#{tag_line}")
    else:
        print(f"Player {game_name}#{tag_line} not found")


def seed_riot_ids(riot_ids):
    for riot_id in riot_ids:
        if "#" not in riot_id:
            print(f"Skipping invalid Riot ID '{riot_id}' (expected name#tag)")
            continue
        name, tag = riot_id.split("#", 1)
        seed_single(name, tag)


def bump_players_by_rank(min_tier="DIAMOND", limit=500):
    min_tier = (min_tier or "DIAMOND").upper()
    min_rank = TIER_ORDER.get(min_tier)
    if min_rank is None:
        raise ValueError(f"Unknown tier: {min_tier}")

    init_db()
    seed = agent.connect()
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT puuid, rank_tier, lp
        FROM players
        WHERE rank_tier IS NOT NULL
        ORDER BY lp DESC
        """
    ).fetchall()
    conn.close()

    bumped = 0
    for row in rows:
        tier = (row["rank_tier"] or "").upper()
        if TIER_ORDER.get(tier, 0) < min_rank:
            continue
        puuid = row["puuid"]
        if seed.sismember("player_handled", puuid) or seed.sismember("player_processing", puuid):
            continue
        seed.zincrby("player_queue", _tier_priority(tier, row["lp"]), puuid)
        bumped += 1
        if bumped >= limit:
            break
    return bumped


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'top':
        seed_top_players()
    elif len(sys.argv) > 1 and '#' in sys.argv[1]:
        name, tag = sys.argv[1].split('#', 1)
        seed_single(name, tag)
    else:
        seed_single("chaos", "oda")
