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
        self.ratelimitcounter = 0   
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
    
    def brpop(self, key, timeout=0):
        return super().brpop(key, timeout=timeout)[1]

    def handle_match(self):
        match = self.get_match_by_id(self.match)
        participants = [match['info']['participants'][i]['puuid'] for i in range(10)]
        #get the players in the match
        self.sadd(f"players_{self.match}", *participants)
        #incr all the players in the match
        #this ups their priority in the check queue
        self.zincrby("player_queue", 1, *participants)
        self.zincrby("players", 1, *participants)
        
        self.lpush("match_import", self.match)
        json.dump(match, open(f"../matches/{self.match}.json", "w"))
        self.srem("match_processing", self.match)
        self.sadd("match_processed", self.match)

    def get_player(self):
        self.player = self.brpop("players").decode("utf-8")
        self.sadd("player_processing", self.player)
        return self.player

    def handle_player(self):
        self.lpush("matches", *self.get_matches_by_puuid(self.player))
        self.srem("player_processing", self.player)

    def get_match(self):
        self.match = self.brpop("matches").decode("utf-8")
        self.sadd("match_processing", self.match)
        return self.match

    def match_import(self, match_id):
        pass

    def ratelimit(self, func, *args, **kwargs):
        endpoint = kwargs.pop("endpoint")
        counter_max = int(self.get(f"cmax_{endpoint}"))
        interval = int(self.get(f"interval_{endpoint}"))
        counter = int(self.get(f"counter{endpoint}"))
        self.incr(f"counter{endpoint}")
        self.expire(f"counter{endpoint}", interval)
        self.ratelimitcounter += 1
        assert self.ratelimitcounter <= 200, (counter, counter_max)
        if counter < counter_max:
            return func(*args, **kwargs)
        else: 
            time.sleep(10*interval/counter_max)
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
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
	        return response.json()
        else:
            assert False, (response)
            return None
    
    def get_match_by_id(self, match_id):
        endpoint = "MATCHV5"
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            assert False, (response)
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
        if response.status_code == 200:
        #parse the response
            return json.loads(response.json())['info']['participants']
        else:
            assert False
            return None


if __name__ == "__main__":
    redis_client = agent.connect()
    for k in redis_client.keys("sec*"):
        redis_client.delete(k)
    redis_client.lpush("players", "wiFvhmOQlgki5o5IfTifgk8wEYdpf0GE2Dw87vU-CGQjBNL6VwpibC8YUgpWVhq0ki0M-8P30J80UA")
    redis_client.delete("cmax_/lol/summoner/v4/summoners/by-account/{encryptedAccountId}")
