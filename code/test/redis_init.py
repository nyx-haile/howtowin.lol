#set up all of the queues for the redis server

import redis

with open('../secrets/howl-fetch', 'r') as f:
    dragonfly_uri = f.read()
    redis_client = redis.from_url(dragonfly_uri)

#set up the queues
#player queue
redis_client.#create list "player_queue"
redis_client.#create list "processing_player_queue"


#game queue
redis_client.#create list "game_queue"
redis_client.#create list "processing_game_queue"

#set up the agents
#do this in build_match_db.py
def player_agent():
    brpoplpush("player_queue", "processing_player_queue", timeout=0)


def game_agent():
    brpoplpush("gq", "pgq", timeout=0)
