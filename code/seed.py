import json
import os
import requests
import sys
import time

from fetch import agent, route_for_platform
from db import get_conn, init_db, insert_player

SEED_CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'seed_cache.json')
SEED_CACHE_MAX_AGE_S = 7 * 24 * 3600  # refresh weekly
RESEED_COOLDOWN_S = 24 * 3600  # per-puuid re-queue cooldown — skip if seen < 24h

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
    reseeded = 0
    skipped_cooldown = 0
    now = time.time()
    for entry in all_entries:
        puuid = entry["puuid"]
        insert_player(conn, puuid, None, entry["tier"], entry["division"], entry["lp"])
        # Always re-queue top players — handle_player is idempotent via
        # sdiff on player_matches_{puuid}, and re-processing picks up newly
        # played games. Without this, small-seed routes (asia: kr+jp1 × 100)
        # go cold permanently after one full drain into player_handled.
        # Cap to once per RESEED_COOLDOWN_S per puuid to avoid wasting API.
        if seed.sismember("player_processing", puuid):
            count += 1
            continue
        last_ts_raw = seed.hget("player_last_reseed", puuid)
        if last_ts_raw is not None:
            try:
                last_ts = float(last_ts_raw)
            except (TypeError, ValueError):
                last_ts = 0.0
            if now - last_ts < RESEED_COOLDOWN_S:
                skipped_cooldown += 1
                count += 1
                continue
        route = route_for_platform(entry["region"])
        seed.zincrby(f"player_queue:{route}", _tier_priority(entry["tier"], entry["lp"]), puuid)
        seed.hset("player_region", puuid, route)
        seed.hset("player_last_reseed", puuid, now)
        reseeded += 1
        count += 1
    print(f"Re-seed stats: queued={reseeded}, skipped_cooldown={skipped_cooldown}, total={count}")

    conn.commit()
    conn.close()
    print(f"Seeded {count} players total")


TIER_DIVISIONS = ["I", "II", "III", "IV"]
LADDER_TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
                "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]


def _fetch_apex_entries(method_name, tier, regions):
    """Call challenger/grandmaster/master endpoint across regions, flatten."""
    out = []
    seed = agent.connect()
    fetch_fn = getattr(seed, method_name)
    for region in _normalize_regions(regions):
        try:
            league = fetch_fn(region=region)
        except requests.exceptions.ConnectionError as e:
            print(f"Skipping {tier} for {region}: {e}", flush=True)
            continue
        if not league:
            continue
        for entry in league.get("entries", []):
            if entry.get("puuid"):
                out.append({
                    "puuid": entry["puuid"],
                    "tier": tier,
                    "division": entry.get("rank", "I"),
                    "lp": int(entry.get("leaguePoints", 0) or 0),
                    "region": region,
                })
        print(f"  {tier:<12} {region:<5} entries={len(league.get('entries', []))}", flush=True)
    return out


def _fetch_ladder_entries(tier, regions, pages_per_division=1):
    """league-exp-v4 entries for IRON..DIAMOND across all four divisions."""
    out = []
    seed = agent.connect()
    for region in _normalize_regions(regions):
        for div in TIER_DIVISIONS:
            for page in range(1, pages_per_division + 1):
                try:
                    entries = seed.get_league_exp_entries(
                        tier, div, page=page, region=region
                    )
                except requests.exceptions.ConnectionError as e:
                    print(f"Skipping {tier} {div} {region}: {e}", flush=True)
                    entries = []
                if not entries:
                    break
                kept = 0
                for entry in entries:
                    if entry.get("puuid"):
                        out.append({
                            "puuid": entry["puuid"],
                            "tier": tier,
                            "division": entry.get("rank", div),
                            "lp": int(entry.get("leaguePoints", 0) or 0),
                            "region": region,
                        })
                        kept += 1
                print(f"  {tier:<12} {region:<5} {div} p{page} entries={kept}",
                      flush=True)
    return out


def seed_rank_diverse(per_tier_cap=2500, regions=None, refresh_cache=False,
                     pages_per_division=1):
    """Seed player_queue with even distribution across all 10 ranked tiers.
    For MASTER/GM/CHALL uses league-v4 apex endpoints; for IRON..DIAMOND
    uses league-exp-v4 paginated. Caps per tier after fetch so tiers with
    more raw entries get downsampled; tiers with fewer (apex) get all.
    """
    init_db()
    conn = get_conn()
    seed = agent.connect()

    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                              'seed_cache_diverse.json')
    cache_valid = (
        not refresh_cache
        and regions is None
        and os.path.exists(cache_path)
        and (time.time() - os.path.getmtime(cache_path)) < SEED_CACHE_MAX_AGE_S
    )
    if cache_valid:
        with open(cache_path) as f:
            by_tier = json.load(f)
        print(f"Using diverse seed cache: "
              f"{sum(len(v) for v in by_tier.values())} players", flush=True)
    else:
        print("Fetching rank-diverse seed from Riot API...", flush=True)
        by_tier: dict[str, list[dict]] = {}
        by_tier["CHALLENGER"] = _fetch_apex_entries(
            "get_challenger_league", "CHALLENGER", regions)
        by_tier["GRANDMASTER"] = _fetch_apex_entries(
            "get_grandmaster_league", "GRANDMASTER", regions)
        by_tier["MASTER"] = _fetch_apex_entries(
            "get_master_league", "MASTER", regions)
        for tier in ["DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER",
                     "BRONZE", "IRON"]:
            by_tier[tier] = _fetch_ladder_entries(tier, regions,
                                                   pages_per_division)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(by_tier, f)
        print(f"Cached diverse seed to {cache_path}", flush=True)

    print("\nCap to per-tier target:", flush=True)
    for tier in LADDER_TIERS:
        raw = by_tier.get(tier, [])
        if len(raw) > per_tier_cap:
            # SHA1-deterministic subsample for reproducibility.
            import hashlib
            raw = sorted(raw, key=lambda e: hashlib.sha1(
                e["puuid"].encode()).hexdigest())[:per_tier_cap]
        by_tier[tier] = raw
        print(f"  {tier:<12} kept={len(raw)}", flush=True)

    count = 0
    reseeded = 0
    skipped_cooldown = 0
    now = time.time()
    for tier in LADDER_TIERS:
        for entry in by_tier.get(tier, []):
            puuid = entry["puuid"]
            insert_player(conn, puuid, None, entry["tier"], entry["division"],
                          entry["lp"])
            if seed.sismember("player_processing", puuid):
                count += 1
                continue
            last_ts_raw = seed.hget("player_last_reseed", puuid)
            if last_ts_raw is not None:
                try:
                    last_ts = float(last_ts_raw)
                except (TypeError, ValueError):
                    last_ts = 0.0
                if now - last_ts < RESEED_COOLDOWN_S:
                    skipped_cooldown += 1
                    count += 1
                    continue
            route = route_for_platform(entry["region"])
            seed.zincrby(f"player_queue:{route}",
                        _tier_priority(entry["tier"], entry["lp"]), puuid)
            seed.hset("player_region", puuid, route)
            seed.hset("player_last_reseed", puuid, now)
            reseeded += 1
            count += 1
    conn.commit()
    conn.close()
    print(f"\nRank-diverse seed: queued={reseeded} "
          f"skipped_cooldown={skipped_cooldown} total={count}", flush=True)


def seed_single(game_name, tag_line):
    init_db()
    seed = agent.connect()
    account = seed.get_account_by_riot_id(game_name, tag_line)
    if account:
        seed.zadd("player_queue:americas", {account["puuid"]: 0})
        seed.hset("player_region", account["puuid"], "americas")
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
        route = (seed.hget("player_region", puuid) or b"americas").decode()
        seed.zincrby(f"player_queue:{route}", _tier_priority(tier, row["lp"]), puuid)
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
