#Helper functions to fetch data from the Riot Games API
import threading
import sqlite3
import requests
import json
import dragon
from redis import Redis
import redis
import time
import random

class agent(Redis):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.id = self.kwargs.get("id", "0")
        self.queue = 420
        self.match_type = self.kwargs.get("match_type", "ranked")
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
        self.set("log", f"{message}\t {self.id}")

    def get_player(self):
        self.log("Getting player")
        self.player = self.bzpopmax("player_queue")[1].decode("utf-8")
        self.log(f"Got player {self.player}")
        self.sadd("player_processing", self.player)
        return self.player

    def zincrby(self, key, increment, *args):
        for arg in args:
            super().zincrby(key, increment, arg)

    def handle_player(self, match_count=None):
        if self.player == None:
            self.get_player()
        if match_count is None:
            match_count = int(self.kwargs.get('match_count', 10))
        matches = self.get_matches_by_puuid(self.player, count=match_count)
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
        endpoint = kwargs.pop("endpoint", "GLOBAL")
        app_prefix = "ratelimit:APP"
        method_prefix = f"ratelimit:{endpoint}"

        def load_windows(prefix, fallback):
            windows = []
            for wid in (1, 2):
                cmax = int(self.get(f"{prefix}:w{wid}:cmax", 0))
                interval = int(self.get(f"{prefix}:w{wid}:interval", 0))
                if cmax > 0 and interval > 0:
                    windows.append((wid, cmax, interval))
            if windows:
                return windows
            return fallback

        app_windows = load_windows(
            app_prefix,
            [
                (1, int(self.get("cmax_short", 20)), int(self.get("interval_short", 1))),
                (2, int(self.get("cmax_long", 100)), int(self.get("interval_long", 120))),
            ],
        )
        method_windows = load_windows(
            method_prefix,
            [
                (1, int(self.get(f"{method_prefix}:cmax_short", 0)), int(self.get(f"{method_prefix}:interval_short", 0))),
                (2, int(self.get(f"{method_prefix}:cmax_long", 0)), int(self.get(f"{method_prefix}:interval_long", 0))),
            ],
        )
        method_windows = [(wid, cmax, interval) for wid, cmax, interval in method_windows if cmax > 0 and interval > 0]

        self.ratelimitcounter += 1
        assert self.ratelimitcounter <= 10000

        while True:
            waits = []

            for wid, cmax, interval in app_windows:
                counter_key = f"{app_prefix}:w{wid}:counter"
                count = int(self.get(counter_key))
                if count >= cmax:
                    waits.append(interval / cmax)

            for wid, cmax, interval in method_windows:
                counter_key = f"{method_prefix}:w{wid}:counter"
                count = int(self.get(counter_key))
                if count >= cmax:
                    waits.append(interval / cmax)

            if not waits:
                for wid, _, interval in app_windows:
                    counter_key = f"{app_prefix}:w{wid}:counter"
                    self.incr(counter_key)
                    self.expire(counter_key, interval)
                for wid, _, interval in method_windows:
                    counter_key = f"{method_prefix}:w{wid}:counter"
                    self.incr(counter_key)
                    self.expire(counter_key, interval)
                return func(*args, **kwargs)

            wait = max(max(waits), 0.2)
            time.sleep(wait)


    def request(self, url, headers, endpoint):
        backoff = 0.5
        max_backoff = 30.0
        max_retries = 8
        attempt = 0

        while True:
            response = self.ratelimit(requests.get, url, headers=headers, endpoint=endpoint)
            if response.status_code != 429:
                return response

            if attempt >= max_retries:
                return response

            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = min(backoff * (2 ** attempt), max_backoff)
            else:
                delay = min(backoff * (2 ** attempt), max_backoff)

            # Spread retries from concurrent workers to avoid synchronized bursts.
            delay += random.uniform(0, min(1.0, delay * 0.25))
            time.sleep(delay)
            attempt += 1

    #Get the matches of a player by PUUID
    def get_matches_by_puuid(self, puuid, start=0, count=100):
        endpoint = "MATCHV5"
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}&queue={self.queue}"
        headers = {
            "X-Riot-Token": self.api_key,
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
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            assert False, (response, '\n', response.json())
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

    def get_challenger_league(self, queue='RANKED_SOLO_5x5', region='na1'):
        endpoint = "LEAGUEV4"
        url = f"https://{region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        return None

    def get_grandmaster_league(self, queue='RANKED_SOLO_5x5', region='na1'):
        endpoint = "LEAGUEV4"
        url = f"https://{region}.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/{queue}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        return None

    def get_summoner_by_id(self, summoner_id, region='na1'):
        endpoint = "SUMMONERV4"
        url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint)
        if response.status_code == 200:
            return response.json()
        return None

#if __name__ == "__main__":
