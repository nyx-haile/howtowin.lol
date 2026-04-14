import time
import _thread
from parser import parser
from fetch import agent
import sys


def player_scraper(x):
    player = agent.connect()
    while player.get_player():
        player.handle_player()


def match_scraper(x):
    match = parser.connect()
    while match.get_match():
        match.handle_match()


if __name__ == "__main__":
    db = agent.connect()
    threads = {'player': {}, 'match': {}}
    num_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i in range(num_threads):
        threads['player'][i] = _thread.start_new_thread(player_scraper, (i,))
        threads['match'][i] = _thread.start_new_thread(match_scraper, (i,))
    print(threads)
    while True:
        time.sleep(0.5)
        print(db.get("log"))
