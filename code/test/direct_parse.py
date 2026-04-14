"""Option A: directly parse real matches end-to-end without the queue."""
import sys
import os
import requests
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parser
from db import get_conn, DEFAULT_DB_PATH
from raw_db import get_raw_conn, DEFAULT_RAW_DB_PATH
from features import compute_feature_vectors

PUUID = 'NJJ-6yMiYjts0beV5FLXaBFjfT8rjBqCqqftmAuHmZnQnKRQDijKPOr7NgfvBd1ffK8NyfkZfus0cw'
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 5

SECRETS = os.path.join(os.path.dirname(__file__), '..', '..', 'secrets')
with open(os.path.join(SECRETS, 'api_key')) as f:
    API_KEY = f.read().strip()
H = {'X-Riot-Token': API_KEY}


def fetch_json(url):
    r = requests.get(url, headers=H)
    r.raise_for_status()
    return r.json()


def main():
    print(f'Fetching {COUNT} matches for chaos#oda')
    match_ids = fetch_json(
        f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{PUUID}/ids?start=0&count={COUNT}'
    )
    print(f'Got: {match_ids}')

    p = parser.connect()

    for i, mid in enumerate(match_ids, 1):
        print(f'\n[{i}/{len(match_ids)}] {mid}')
        m = fetch_json(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}')
        tl = fetch_json(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}/timeline')
        p.match = mid
        p.handle_match(match_data=m, match_timeline=tl)
        time.sleep(0.3)

    print('\n=== DB counts ===')
    conn = get_conn()
    for table in ['games', 'players', 'frames', 'events']:
        n = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print(f'  {table}: {n}')
    conn.close()

    raw = get_raw_conn()
    n_raw = raw.execute('SELECT COUNT(*) FROM raw_matches').fetchone()[0]
    sz = raw.execute('SELECT SUM(LENGTH(match_json) + LENGTH(timeline_json)) FROM raw_matches').fetchone()[0]
    print(f'  raw_matches: {n_raw} rows, {sz/1024:.0f} KB total')
    raw.close()

    print('\n=== feature vector sample ===')
    sample = compute_feature_vectors(match_ids[0])
    if sample:
        f0 = sample[0]
        print(f'  team=100 at t=0: {dict(list(f0.items())[:8])}')
        print(f'  total vectors for {match_ids[0]}: {len(sample)}')
    else:
        print('  (none)')

    print(f'\nDB: {os.path.abspath(DEFAULT_DB_PATH)}')
    print(f'Raw: {os.path.abspath(DEFAULT_RAW_DB_PATH)}')


if __name__ == '__main__':
    main()
