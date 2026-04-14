"""Seed the Redis rate-limit keys the fetcher expects.

Riot dev key limits: 20 req/sec and 100 req/2min globally.
We mirror those per-endpoint with headroom (80 req / 120 sec).
"""
from fetch import agent

ENDPOINTS = ['ACCOUNTV1', 'MATCHV5', 'LEAGUEV4', 'SUMMONERV4', 'SPECTATORV4']

CMAX = 80
INTERVAL = 120


def init_rate_limits():
    r = agent.connect()
    for e in ENDPOINTS:
        r.set(f'cmax_{e}', CMAX)
        r.set(f'interval_{e}', INTERVAL)
    print(f'Seeded rate limits for {len(ENDPOINTS)} endpoints: {CMAX} req / {INTERVAL} sec')


def clear_queues():
    r = agent.connect()
    keys = [
        'player_queue', 'match_queue',
        'player_processing', 'match_processing',
        'player_handled', 'match_handled',
        'players', 'matches',
    ]
    for k in keys:
        r.delete(k)
    # player_matches_* sets
    for k in r.scan_iter(match='player_matches_*'):
        r.delete(k)
    print('Cleared queue and tracking keys')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'clear':
        clear_queues()
    init_rate_limits()
