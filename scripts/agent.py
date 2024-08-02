import time
import _thread
import fetch
from fetch import agent
#import match
import dragon
from dragon import Dragon as dragon
import sys


def player_scraper(x):
    #start player
    player = agent.connect()
    while player.get_player():
        player.handle_player()
    player.set(f"scraper_{x}", "done")

def match_parser(x):
    #start match scraper
    match = agent.connect()
    while match.get_match():
        match.handle_match()

if __name__ == "__main__":
    #start player parser agent
    db = agent.connect()
    threads = {'player': {}, 'match': {}}
    for i in range(int(sys.argv[1])):
        #threads['player'][i] = _thread.start_new_thread(player_scraper, (i,))
        threads['match'][i] = _thread.start_new_thread(match_parser, (i,))
    while db.get("scraper_0") != "done": 
        time.sleep(2)
    print(db.get("scraper_0"))



