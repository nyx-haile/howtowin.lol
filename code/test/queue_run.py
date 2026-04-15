"""Option B: drive the real queue path end-to-end with a bounded runtime.

Mirrors agents.py but uses timeouts on the blocking pops so we can stop once
the queue drains.
"""
import sys
import os
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch import agent
from parser import parser
from db import get_conn

IDLE_TIMEOUT_S = 15
MAX_WALL_S = 4800
MATCH_COUNT = 10
MON_INTERVAL_S = 15


def drain_players(stop_flag):
    try:
        a = agent.connect()
        print('[player] connected', flush=True)
    except Exception:
        print('[player] CONNECT FAILED', flush=True)
        traceback.print_exc()
        return
    while not stop_flag[0]:
        try:
            res = a.bzpopmax('player_queue', timeout=IDLE_TIMEOUT_S)
        except Exception:
            print('[player] bzpopmax exception', flush=True)
            traceback.print_exc()
            return
        if res is None:
            print('[player] queue idle, exiting', flush=True)
            return
        a.player = res[1].decode('utf-8')
        a.sadd('player_processing', a.player)
        print(f'[player] handling {a.player[:20]}...', flush=True)
        try:
            a.handle_player(match_count=MATCH_COUNT)
            print(f'[player] done {a.player[:20]}', flush=True)
        except Exception:
            print('[player] handle_player error', flush=True)
            traceback.print_exc()


def drain_matches(stop_flag):
    try:
        p = parser.connect()
        print('[match] connected', flush=True)
    except Exception:
        print('[match] CONNECT FAILED', flush=True)
        traceback.print_exc()
        return
    while not stop_flag[0]:
        try:
            res = p.bzpopmax('match_queue', timeout=IDLE_TIMEOUT_S)
        except Exception:
            print('[match] bzpopmax exception', flush=True)
            traceback.print_exc()
            return
        if res is None:
            pq = p.zcard('player_queue')
            pp = p.scard('player_processing')
            if pq > 0 or pp > 0:
                print(f'[match] empty but players still working (pq={pq}, proc={pp}), waiting...', flush=True)
                continue
            print('[match] queue idle, exiting', flush=True)
            return
        p.match = res[1].decode('utf-8')
        p.sadd('match_processing', p.match)
        print(f'[match] parsing {p.match}', flush=True)
        try:
            p.handle_match()
            p.srem('match_processing', p.match)
            p.sadd('match_handled', p.match)
            print(f'[match] done {p.match}', flush=True)
        except Exception:
            print(f'[match] handle_match error on {p.match}', flush=True)
            traceback.print_exc()


def main():
    stop = [False]
    t1 = threading.Thread(target=drain_players, args=(stop,), daemon=True)
    t2 = threading.Thread(target=drain_matches, args=(stop,), daemon=True)
    t1.start()
    t2.start()

    mon = agent.connect()
    start = time.time()
    while time.time() - start < MAX_WALL_S:
        time.sleep(MON_INTERVAL_S)
        pq = mon.zcard('player_queue')
        mq = mon.zcard('match_queue')
        pp = mon.scard('player_processing')
        mp = mon.scard('match_processing')
        handled = mon.scard('match_handled')
        conn = get_conn()
        games = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        frames = conn.execute('SELECT COUNT(*) FROM frames').fetchone()[0]
        events = conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        conn.close()
        elapsed = int(time.time() - start)
        print(f'  [mon t={elapsed}s] pq={pq}/{pp} mq={mq}/{mp} handled={handled} | games={games} frames={frames} events={events}', flush=True)
        if pq == 0 and mq == 0 and pp == 0 and mp == 0 and handled > 0:
            print('[mon] drained, stopping')
            break

    stop[0] = True
    time.sleep(IDLE_TIMEOUT_S + 1)
    print('Done')


if __name__ == '__main__':
    main()
