#Helper functions to fetch data from the Riot Games API
import sqlite3
import requests
import json
import dragon
from redis import Redis as redis
import time

class Agent(redis):
    def get(self, key, default=0):
        val = super().get(key)
        if val == None:
            return default
        return val

def redis_connect():
    with open("../secrets/howl-fetch", "r") as file:
        dragonfly_uri = file.read()
        redis_client = Agent.from_url(dragonfly_uri)
    return redis_client

#API key
#fetch from secrets directory
with open("../secrets/api_key", "r") as file:
    api_key = file.read().strip()
    print(type(api_key))
    print(api_key)
#define ratelimited request
def request(url, headers, endpoint):
    return ratelimit(requests.get, url, headers=headers, endpoint=endpoint)

#Get the matches of a player by PUUID
def get_matches_by_puuid(puuid, match_type="ranked", start=0, count=100):
    endpoint = "/lol/match/v5/matches/by-puuid/{puuid}/ids"
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
    headers = {
	    "X-Riot-Token": api_key,
        "type" : match_type
    }
    print(endpoint)
    response = request(url, headers=headers, endpoint=endpoint)
    if response.status_code == 200:
	    return response.json()
    else:
	    return None

def get_timeline_by_match(match_id):
    endpoint = "/lol/match/v5/matches/{matchId}/timeline"
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {
        "X-Riot-Token": api_key,
    }
    response = request(url, headers=headers, endpoint=endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        return None

#check the other players in each game
def get_players_by_match(match_id):
    endpoint = "/lol/match/v5/matches/{matchId}"
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {
	    "X-Riot-Token": api_key
    }
    response = request(url, headers=headers, endpoint=endpoint)
    print(response)
    if response.status_code == 200:
	#parse the response
        handle_match(response.json())
    else:
        assert False
        return None

#can lpush(key *vals) to add multiple values to a list

def handle_match(match):
    #get the players in the match
    participants = match["info"]["participants"]
    redis_client.lpush("players", *[participant["puuid"] for participant in participants])
    redis_client.lpush("match_import", match["metadata"]["matchId"])

def handle_player(player):
    redis_client.lpush("matches", *get_matches_by_puuid(player))
    
def match_import(match_id):
    pass

def ratelimit(func, *args, **kwargs):
      endpoint = kwargs["endpoint"]
      redis_client = redis_connect()
      counter_max = redis_client.get(f"cmax_{endpoint}")
      interval = redis_client.get(f"interval_{endpoint}")
      counter = redis_client.get(f"counter{endpoint}")
      redis_client.incr(f"counter{endpoint}")
      redis_client.expire(f"counter{endpoint}", interval)
      if counter < counter_max:
          return func(*args, **kwargs)
      else: 
          time.sleep(interval/counter_max)
          return func(*args, **kwargs)


if __name__ == "__main__":
    redis_client = redis_connect()
    redis_client.lpush("players", "wiFvhmOQlgki5o5IfTifgk8wEYdpf0GE2Dw87vU-CGQjBNL6VwpibC8YUgpWVhq0ki0M-8P30J80UA")
    player = redis_client.rpop("players").decode("utf-8")
    while player != None: #while there are still players in the database
        matches = get_matches_by_puuid(player)
        for match in matches:
            handle_match(match)
        assert False
        #get the next player
