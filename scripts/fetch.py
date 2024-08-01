#Helper functions to fetch data from the Riot Games API

import requests
import json
import dragon
import redis
import time

class Agent():
    def __init__(self):
        self.db = redis_connect()
        self.api_key = api_key()

def redis_connect():
    with open("../secrets/howl-fetch", "r") as file:
        dragonfly_uri = file.read()
        redis_client = redis.from_url(dragonfly_uri)
    return redis_client

#API key
#fetch from secrets directory
def api_key():
    with open("../secrets/api_key.txt", "r") as file:
        api_key = file.read()

#define ratelimited request
def request(url, headers):
    return ratelimit(requests.get, url, headers=headers)

#Get the matches of a player by PUUID
def get_matches_by_puuid(puuid, start=0, count=100):
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
    headers = {
	    "X-Riot-Token": api_key()
        "type" : "ranked"
    }
    response = request(url, headers=headers)
    if response.status_code == 200:
	    return response.json()
    else:
	    return None

#check the other players in each game
def get_players_by_match(match_id, match_type="ranked"):
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {
	    "X-Riot-Token": api_key
	    "type" : match_type
    }
    response = request(url, headers=headers)
    if response.status_code == 200:
	#parse the response
	handle_match(response.json())
    else:
    	return None

def handle_match(match):
    #get the players
    players = match["info"]["participants"]
    for player in players:
	    #check if the player is already in the database
    	#convert to redis
        player_cursor.execute("SELECT * FROM players WHERE puuid=?", (player["puuid"],))
	    if player_cursor.fetchone() == None:
	        #add the player to the database and store the timestamp of their matches
            #may be used later for a prediction algorithm to make checking more efficient.
	        player_cursor.execute("INSERT INTO players (puuid) VALUES (?)", (player["puuid"],))
	        player_db.commit()
	        #get the matches of the player
	        matches = get_matches_by_puuid(player["puuid"])
	        if matches != None:
		        for m in matches:
	    	        handle_match(m)
	        else:
		        print("Failed to get matches for player " + player["puuid"])
        else:
	        print("Player " + player["puuid"] + " is already in the database")

#define ratelimit function which makes sure we don't exceed the rate limit of 100 requests per 2 minutes
def ratelimit(func, *args):
      #define two lists. sec_rq stores the number of requests in the last second (max 20)
      #min_rq stores the number of requests in the last 2 minutes (max 100)
      now = int(time.time())
      ts = now  % 100
      sec_rq = redis_client.get(f"sec_rq{ts}")
      min_rq = redis_client.get(f"min_rq") % 100
      time.sleep(max(redis_client.get(f"min_rq{min_rq}") + 120 - now, 0))
      if sec_rq <= 20:
         redis_client.incr(f"sec_rq{ts}")
         redis_client.expire(f"sec_rq{ts}", now + 1)
         redis_client.incr("min_rq")
         redis_client.set(f"min_rq{min_rq}", now)
         func(*args)
      else:
          time.sleep(1)
          ratelimit(func, *args)

if __name__ == "__main__":
    #set up working loop
    #get first player from player database
    player_cursor.execute("SELECT * FROM players")
    player = player_cursor.fetchone()
    while player != None: #while there are still players in the database
        matches = get_matches_by_puuid(player[0])
        for match in matches:
            handle_match(match)
        #get the next player
