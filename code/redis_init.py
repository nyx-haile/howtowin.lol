"""Seed the Redis rate-limit keys the fetcher expects.

Riot dev key limits are global across endpoints: 20 req/sec and 100 req/2min.
We use two rolling windows with small headroom.
"""
from fetch import agent

CMAX_SHORT = 18
INTERVAL_SHORT = 1
CMAX_LONG = 95
INTERVAL_LONG = 120


def init_rate_limits():
    r = agent.connect()
    r.set('cmax_short', CMAX_SHORT)
    r.set('interval_short', INTERVAL_SHORT)
    r.set('cmax_long', CMAX_LONG)
    r.set('interval_long', INTERVAL_LONG)
    r.delete('counter_short', 'counter_long')
    print(f'Seeded global rate limits: {CMAX_SHORT}/{INTERVAL_SHORT}s + {CMAX_LONG}/{INTERVAL_LONG}s')


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
