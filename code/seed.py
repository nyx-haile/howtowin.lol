from fetch import agent
from db import get_conn, init_db, insert_player
import sys


def seed_top_players(per_tier=50):
    init_db()
    seed = agent.connect()
    conn = get_conn()
    count = 0

    for tier, fetch_fn in [('CHALLENGER', seed.get_challenger_league),
                           ('GRANDMASTER', seed.get_grandmaster_league)]:
        league = fetch_fn()
        if not league:
            print(f"Failed to fetch {tier} league")
            continue

        entries = sorted(league.get('entries', []),
                         key=lambda e: e.get('leaguePoints', 0), reverse=True)[:per_tier]

        for entry in entries:
            puuid = entry.get('puuid')
            if not puuid:
                continue
            lp = entry.get('leaguePoints', 0)
            insert_player(conn, puuid, None, tier, entry.get('rank', 'I'), lp)
            seed.zadd("player_queue", {puuid: lp})
            count += 1

        print(f"Seeded {per_tier} from {tier}")

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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'top':
        seed_top_players()
    elif len(sys.argv) > 1 and '#' in sys.argv[1]:
        name, tag = sys.argv[1].split('#', 1)
        seed_single(name, tag)
    else:
        seed_single("chaos", "oda")
