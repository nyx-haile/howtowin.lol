"""Seed Redis rate-limit keys the fetcher expects."""
from fetch import agent

APP_LIMITS = {
    'w1': {'cmax': 20, 'interval': 1},
    'w2': {'cmax': 100, 'interval': 120},
}

METHOD_LIMITS = {
    # From observed response headers.
    'MATCHV5': {
        'w1': {'cmax': 2000, 'interval': 10},
    },
    'ACCOUNTV1': {
        'w1': {'cmax': 1000, 'interval': 60},
    },
    'LEAGUEV4': {
        'w1': {'cmax': 30, 'interval': 10},
        'w2': {'cmax': 500, 'interval': 600},
    },
    # Not yet probed in this session; keep conservative default.
    'SUMMONERV4': {
        'w1': {'cmax': 1000, 'interval': 60},
    },
}


def init_rate_limits():
    r = agent.connect()

    # Legacy keys for fallback compatibility.
    r.set('cmax_short', APP_LIMITS['w1']['cmax'])
    r.set('interval_short', APP_LIMITS['w1']['interval'])
    r.set('cmax_long', APP_LIMITS['w2']['cmax'])
    r.set('interval_long', APP_LIMITS['w2']['interval'])
    r.delete('counter_short', 'counter_long')

    # Clear stale per-route APP counters from previous runs.
    for key in r.scan_iter('ratelimit:APP:*:counter'):
        r.delete(key)

    app_prefix = 'ratelimit:APP'
    for wid, spec in APP_LIMITS.items():
        r.set(f'{app_prefix}:{wid}:cmax', spec['cmax'])
        r.set(f'{app_prefix}:{wid}:interval', spec['interval'])
        r.delete(f'{app_prefix}:{wid}:counter')

    for endpoint, windows in METHOD_LIMITS.items():
        prefix = f'ratelimit:{endpoint}'
        for wid in ('w1', 'w2'):
            spec = windows.get(wid)
            if spec:
                r.set(f'{prefix}:{wid}:cmax', spec['cmax'])
                r.set(f'{prefix}:{wid}:interval', spec['interval'])
            else:
                r.set(f'{prefix}:{wid}:cmax', 0)
                r.set(f'{prefix}:{wid}:interval', 0)
            r.delete(f'{prefix}:{wid}:counter')

    print('Seeded rate limits:')
    print(
        f"  APP: {APP_LIMITS['w1']['cmax']}/{APP_LIMITS['w1']['interval']}s "
        f"+ {APP_LIMITS['w2']['cmax']}/{APP_LIMITS['w2']['interval']}s"
    )
    for endpoint, windows in METHOD_LIMITS.items():
        parts = []
        for wid in ('w1', 'w2'):
            spec = windows.get(wid)
            if spec:
                parts.append(f"{spec['cmax']}/{spec['interval']}s")
        print(
            f"  {endpoint}: {' + '.join(parts) if parts else 'disabled'}"
        )


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
