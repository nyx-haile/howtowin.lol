#Helper functions to fetch data from the Riot Games API
import threading
import sqlite3
import requests
import json
import dragon
from redis import Redis
import redis
import time

class agent(Redis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_type = "ranked"
        with open("../secrets/api_key", "r") as file:
            self.api_key = file.read().strip()
    
    @classmethod
    def connect(cls):
        with open("../secrets/howl-fetch", "r") as file:
            uri = file.read()
        return cls.from_url(uri)

    def get(self, key, default=0):
        val = super().get(key)
        if val == None:
            return default
        return val

    def handle_match(match):
        print(match)
        match = json.loads(match)
        #get the players in the match
        participants = match["info"]["participants"]
        self.db.lpush("players", *[participant["puuid"] for participant in participants])
        self.db.lpush("match_import", match["metadata"]["matchId"])

    def handle_player(self, player):
        self.lpush("matches", *self.get_matches_by_puuid(player, self.api_key))
    
    def match_import(self, match_id):
        pass

    def ratelimit(self, func, *args, **kwargs):
        endpoint = kwargs.pop("endpoint")
        counter_max = int(self.get(f"cmax_{endpoint}"))
        interval = int(self.get(f"interval_{endpoint}"))
        counter = int(self.get(f"counter{endpoint}"))
        self.incr(f"counter{endpoint}")
        self.expire(f"counter{endpoint}", interval)
        if counter < counter_max:
            return func(*args, **kwargs)
        else: 
            time.sleep(interval/counter_max)
            return func(*args, **kwargs)


    def request(self, url, headers, endpoint):
        return self.ratelimit(requests.get, url, headers=headers, endpoint=endpoint)

    #Get the matches of a player by PUUID
    def get_matches_by_puuid(self, puuid, start=0, count=100):
        endpoint = "MATCHV5"
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
        headers = {
	        "X-Riot-Token": self.api_key,
            "type" : self.match_type
        }
        print(endpoint)
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
	        return response.json()
        else:
	        return None

    def get_timeline_by_match(self, match_id):
        endpoint = "MATCHV5"
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        headers = {
            "X-Riot-Token": api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    #check the other players in each game
    def get_players_by_match(match_id):
        endpoint = "MATCHV5"
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
        headers = {
            "X-Riot-Token": api_key
        }
        response = self.request(url, headers=headers, endpoint=endpoint)
        print(response)
        if response.status_code == 200:
        #parse the response
            handle_match(response.json())
        else:
            assert False
            return None


if __name__ == "__main__":
    redis_client = agent.connect()
    redis_client.lpush("players", "wiFvhmOQlgki5o5IfTifgk8wEYdpf0GE2Dw87vU-CGQjBNL6VwpibC8YUgpWVhq0ki0M-8P30J80UA")
    player = redis_client.rpop("players").decode("utf-8")
    redis_client.set("test", "hello world")
    print(redis_client.get("test"))
    print(redis_client.get("tes3443t", 69))
    print(redis_client.keys())
    print(redis_client.get("interval_MATCHV5", 69))
    redis_client.handle_player(player)
