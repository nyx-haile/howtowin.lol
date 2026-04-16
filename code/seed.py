import requests
import sys

from fetch import agent
from db import get_conn, init_db, insert_player

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


def seed_top_players(per_tier=50, regions=None, include_grandmaster=True):
    init_db()
    seed = agent.connect()
    conn = get_conn()
    count = 0

    tiers = [("CHALLENGER", seed.get_challenger_league)]
    if include_grandmaster:
        tiers.append(("GRANDMASTER", seed.get_grandmaster_league))

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

            seeded = 0
            for entry in entries:
                puuid = entry.get("puuid")
                if not puuid:
                    continue
                lp = int(entry.get("leaguePoints", 0) or 0)
                division = entry.get("rank", "I")
                insert_player(conn, puuid, None, tier, division, lp)
                if not seed.sismember("player_handled", puuid):
                    seed.zincrby("player_queue", _tier_priority(tier, lp), puuid)
                seeded += 1
                count += 1

            print(f"Seeded {seeded} from {tier} ({region})")

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
