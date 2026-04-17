"""Redis rate-limit helpers. Rate limiting is now 429-driven — no local counters."""
from fetch import agent


def init_rate_limits():
    r = agent.connect()
    cleared = 0
    for key in r.scan_iter('ratelimit:*'):
        r.delete(key)
        cleared += 1
    if cleared:
        print(f'Cleared {cleared} stale rate-limit keys')
    else:
        print('Rate limits clean')


def clear_queues():
    r = agent.connect()
    for k in ['player_processing', 'match_processing',
              'player_handled', 'match_handled',
              'player_queue', 'match_queue',
              'players', 'matches']:
        r.delete(k)
    for pattern in ('player_queue:*', 'match_queue:*', 'player_matches_*'):
        for k in r.scan_iter(match=pattern):
            r.delete(k)
    print('Cleared queue and tracking keys')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'clear':
        clear_queues()
    init_rate_limits()
