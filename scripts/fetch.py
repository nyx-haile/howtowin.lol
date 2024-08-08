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
        if self.match == None:
            self.get_match()
        match = self.get_match_by_id(self.match)
        participants = [i['puuid'] for i in match['info']['participants']]
        #get the players in the match
        #incr all the players in the match
        #this ups their priority in the check queue
        self.zincrby("player_queue", 1, *participants)
        #save the data to a file
        json.dump(match, open(f"../matches/{self.match}.json", "w"))
        #remove the match from the processing queue
        self.srem("match_processing", self.match)
        self.sadd("match_handled", self.match)
        self.zincrby("matches", 1, self.match)

    def log(self, message):
        self.set("log", message)
    def get_player(self):
        self.log("Getting player")
        self.player = self.bzpopmax("player_queue")[1].decode("utf-8")
        self.log(f"Got player {self.player}")
        self.sadd("player_processing", self.player)
        return self.player

    def zincrby(self, key, increment, *args):
        for arg in args:
            super().zincrby(key, increment, arg)

    def handle_player(self):
        if self.player == None:
            self.get_player()
        matches = self.get_matches_by_puuid(self.player)
        self.set("log", f"Player {self.player} has {len(matches)} matches")
        player_data = self.get_account_by_puuid(self.player)
        self.set("log", f"Player {self.player} is {player_data}")
        #incr all the matches not already handled
        self.set("log", "dumping matches")
        self.sadd(f"player_matches_{self.player}", *matches)
        new_matches = self.sdiff(f"player_matches_{self.player}", "match_handled", "match_processing")
        nmc = self.scard(f"player_matches_{self.player}")
        self.zincrby("match_queue", 1, *new_matches)
        self.srem("player_processing", self.player)
        self.sadd("player_handled", self.player)
        self.zincrby("players", nmc, self.player)


    def get_match(self):
        self.match = self.bzpopmax("match_queue")[1].decode("utf-8")
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
            assert False, (response, response.json())
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

    def get_account_by_puuid(self, puuid):
        endpoint = "ACCOUNTV1"
        url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        else:
           assert False, (response, response.json())
           return response


    def get_account_by_riot_id(self, gameName, tagLine):
        endpoint = "ACCOUNTV1"
        url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine}"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            return None

if __name__ == "__main__":
    redis_client = agent.connect()
