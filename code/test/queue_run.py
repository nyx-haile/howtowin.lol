"""Per-route queue worker — drives player_queue:{route} and match_queue:{route}."""
import sys
import os
import argparse
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch import agent
from parser import parser
from db import get_conn

IDLE_TIMEOUT_S = 15
MAX_WALL_S = int(os.environ.get("HOWL_QUEUE_MAX_WALL_S", 28800))  # default 8h; grow_games manages termination on target
MATCH_COUNT = 100
MON_INTERVAL_S = 15


def drain_players(route, stop_flag):
    pq = f'player_queue:{route}'
    try:
        a = agent.connect()
        print(f'[player/{route}] connected', flush=True)
    except Exception:
        print(f'[player/{route}] CONNECT FAILED', flush=True)
        traceback.print_exc()
        return
    while not stop_flag[0]:
        try:
            res = a.bzpopmax(pq, timeout=IDLE_TIMEOUT_S)
        except Exception:
            print(f'[player/{route}] bzpopmax exception', flush=True)
            traceback.print_exc()
            return
        if res is None:
            print(f'[player/{route}] queue idle, exiting', flush=True)
            return
        a.player = res[1].decode('utf-8')
        a.sadd('player_processing', a.player)
        t0 = time.time()
        try:
            a.handle_player(match_count=MATCH_COUNT, skip_rank=False, route=route)
            print(f'[player/{route}] done {a.player[:20]} in {time.time()-t0:.1f}s', flush=True)
        except Exception:
            print(f'[player/{route}] handle_player error after {time.time()-t0:.1f}s', flush=True)
            traceback.print_exc()


def drain_matches(route, stop_flag):
    mq = f'match_queue:{route}'
    try:
        p = parser.connect()
        print(f'[match/{route}] connected', flush=True)
    except Exception:
        print(f'[match/{route}] CONNECT FAILED', flush=True)
        traceback.print_exc()
        return
    while not stop_flag[0]:
        try:
            res = p.bzpopmax(mq, timeout=IDLE_TIMEOUT_S)
        except Exception:
            print(f'[match/{route}] bzpopmax exception', flush=True)
            traceback.print_exc()
            return
        if res is None:
            remaining_pq = p.zcard(f'player_queue:{route}')
            if remaining_pq > 0:
                print(f'[match/{route}] empty but player queue has {remaining_pq}, waiting...', flush=True)
                continue
            print(f'[match/{route}] queue idle, exiting', flush=True)
            return
        p.match = res[1].decode('utf-8')
        p.sadd('match_processing', p.match)
        t0 = time.time()
        try:
            p.handle_match()
            p.srem('match_processing', p.match)
            p.sadd('match_handled', p.match)
            print(f'[match/{route}] done {p.match} in {time.time()-t0:.1f}s', flush=True)
        except Exception:
            print(f'[match/{route}] error on {p.match} after {time.time()-t0:.1f}s', flush=True)
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', required=True,
                    choices=['americas', 'europe', 'asia', 'sea'],
                    help='Riot routing region this worker handles')
    args = ap.parse_args()
    route = args.route

    stop = [False]
    t1 = threading.Thread(target=drain_players, args=(route, stop), daemon=True)
    t2 = threading.Thread(target=drain_matches, args=(route, stop), daemon=True)
    t1.start()
    t2.start()

    mon = agent.connect()
    start = time.time()
    while time.time() - start < MAX_WALL_S:
        time.sleep(MON_INTERVAL_S)
        pq = mon.zcard(f'player_queue:{route}')
        mq = mon.zcard(f'match_queue:{route}')
        pp = mon.scard('player_processing')
        mp = mon.scard('match_processing')
        handled = mon.scard('match_handled')
        conn = get_conn()
        games = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        conn.close()
        elapsed = int(time.time() - start)
        print(f'  [mon/{route} t={elapsed}s] pq={pq}/{pp} mq={mq}/{mp} handled={handled} games={games}', flush=True)
        if pq == 0 and mq == 0:
            print(f'[mon/{route}] drained, stopping')
            break

    stop[0] = True
    time.sleep(IDLE_TIMEOUT_S + 1)
    print(f'Done [{route}]')


if __name__ == '__main__':
    main()
