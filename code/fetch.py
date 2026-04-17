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
from concurrent.futures import ThreadPoolExecutor
from db import get_conn, insert_player

# Riot regional routing — rate limits are independent per route.
PLATFORM_TO_ROUTE = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "kr": "asia", "jp1": "asia",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea", "ph2": "sea", "th2": "sea",
}

def route_for_platform(platform):
    return PLATFORM_TO_ROUTE.get(platform.lower(), "americas")

def route_for_match(match_id):
    if "_" in match_id:
        return route_for_platform(match_id.split("_")[0])
    return "americas"

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

    def handle_player(self, match_count=None, skip_rank=False):
        if self.player == None:
            self.get_player()
        if match_count is None:
            match_count = int(self.kwargs.get('match_count', 10))

        # Look up which route this player belongs to.
        player_route = (self.hget("player_region", self.player) or b"americas").decode()

        # Only fetch match list — skip account/rank lookups to save API calls.
        matches = self.get_matches_by_puuid(self.player, count=match_count, route=player_route)
        self.set("log", f"Player {self.player} has {len(matches)} matches")

        rank_tier = None
        rank_division = None
        rank_lp = None

        if not skip_rank:
            platform = "na1"
            if matches:
                first_match = matches[0]
                if "_" in first_match:
                    platform = first_match.split("_", 1)[0].lower()

            player_data = self.get_account_by_puuid(self.player, route=player_route)
            summoner = self.get_summoner_by_puuid(self.player, region=platform)
            if summoner and summoner.get("id"):
                entries = self.get_league_entries_by_summoner(summoner["id"], region=platform)
                if entries:
                    solo = [e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"]
                    if solo:
                        top = max(
                            solo,
                            key=lambda e: (self._tier_order(e.get("tier")), int(e.get("leaguePoints", 0) or 0)),
                        )
                        rank_tier = top.get("tier")
                        rank_division = top.get("rank")
                        rank_lp = int(top.get("leaguePoints", 0) or 0)
        else:
            player_data = None

        riot_id = None
        if player_data and player_data.get("gameName"):
            riot_id = f"{player_data.get('gameName')}#{player_data.get('tagLine', '')}"
        conn = get_conn()
        insert_player(conn, self.player, riot_id, rank_tier, rank_division, rank_lp)
        conn.commit()
        conn.close()

        # incr all the matches not already handled
        self.set("log", "dumping matches")
        nmc = len(matches)
        new_matches = []
        if matches:
            self.sadd(f"player_matches_{self.player}", *matches)
            new_matches = self.sdiff(f"player_matches_{self.player}", "match_handled", "match_processing")
        if new_matches:
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
        route = kwargs.pop("route", "americas")
        # Scope rate-limit counters per route — each Riot routing value has independent limits.
        app_prefix = f"ratelimit:APP:{route}"
        method_prefix = f"ratelimit:{endpoint}:{route}"
        # Read window configs from route-agnostic keys (set by redis_init).
        cfg_app = "ratelimit:APP"
        cfg_method = f"ratelimit:{endpoint}"

        def load_windows(cfg_prefix, fallback):
            windows = []
            for wid in (1, 2):
                cmax = int(self.get(f"{cfg_prefix}:w{wid}:cmax", 0))
                interval = int(self.get(f"{cfg_prefix}:w{wid}:interval", 0))
                if cmax > 0 and interval > 0:
                    windows.append((wid, cmax, interval))
            if windows:
                return windows
            return fallback

        app_windows = load_windows(
            cfg_app,
            [
                (1, int(self.get("cmax_short", 20)), int(self.get("interval_short", 1))),
                (2, int(self.get("cmax_long", 100)), int(self.get("interval_long", 120))),
            ],
        )
        method_windows = load_windows(
            cfg_method,
            [
                (1, int(self.get(f"{cfg_method}:cmax_short", 0)), int(self.get(f"{cfg_method}:interval_short", 0))),
                (2, int(self.get(f"{cfg_method}:cmax_long", 0)), int(self.get(f"{cfg_method}:interval_long", 0))),
            ],
        )
        method_windows = [(wid, cmax, interval) for wid, cmax, interval in method_windows if cmax > 0 and interval > 0]

        all_windows = [(f"{app_prefix}:w{wid}:counter", cmax, interval)
                       for wid, cmax, interval in app_windows]
        all_windows += [(f"{method_prefix}:w{wid}:counter", cmax, interval)
                        for wid, cmax, interval in method_windows]

        while True:
            # Atomically claim a slot in every window. INCR returns the new
            # value so each worker gets a unique count — no read-then-write race.
            # Only set TTL when the key is first created (new_val == 1) so the
            # window expires naturally and we don't reset it on every attempt.
            claimed = []
            over = None
            for key, cmax, interval in all_windows:
                new_val = int(self.incr(key))
                if new_val == 1:
                    self.expire(key, interval)
                claimed.append((key, new_val, cmax, interval))
                if new_val > cmax and over is None:
                    over = (key, new_val, cmax, interval)

            if over is None:
                # All windows have room — make the request.
                return func(*args, **kwargs)

            # We overshot at least one window; roll back all claims and sleep.
            for key, _, _, _ in claimed:
                self.decr(key)
            key, new_val, cmax, interval = over
            # Sleep until the window likely has room — check TTL for accuracy.
            ttl = self.ttl(key)
            wait = max(ttl / max(new_val, 1), 0.1) if ttl > 0 else (interval / cmax)
            time.sleep(wait)


    def request(self, url, headers, endpoint, route="americas"):
        backoff = 0.5
        max_backoff = 30.0
        max_retries = 8
        attempt = 0

        while True:
            response = self.ratelimit(requests.get, url, headers=headers, endpoint=endpoint, route=route)
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
    def get_matches_by_puuid(self, puuid, start=0, count=100, route="americas"):
        endpoint = "MATCHV5"
        url = f"https://{route}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}&queue={self.queue}"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        else:
            assert False, (response, response.json())
            return None

    def get_match_by_id(self, match_id):
        endpoint = "MATCHV5"
        route = route_for_match(match_id)
        url = f"https://{route}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        else:
            assert False, (response)
            return None

    def get_timeline_by_match(self, match_id):
        endpoint = "MATCHV5"
        route = route_for_match(match_id)
        url = f"https://{route}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        else:
            assert False, (response, '\n', response.json())
            return None

    def get_account_by_puuid(self, puuid, route="americas"):
        endpoint = "ACCOUNTV1"
        url = f"https://{route}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        else:
           assert False, (response, response.json())
           return response


    def get_account_by_riot_id(self, gameName, tagLine, route="americas"):
        endpoint = "ACCOUNTV1"
        url = f"https://{route}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine}"
        headers = {
            "X-Riot-Token": self.api_key,
        }
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_challenger_league(self, queue='RANKED_SOLO_5x5', region='na1'):
        endpoint = "LEAGUEV4"
        route = route_for_platform(region)
        url = f"https://{region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        return None

    def get_grandmaster_league(self, queue='RANKED_SOLO_5x5', region='na1'):
        endpoint = "LEAGUEV4"
        route = route_for_platform(region)
        url = f"https://{region}.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/{queue}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        return None

    def get_summoner_by_id(self, summoner_id, region='na1'):
        endpoint = "SUMMONERV4"
        route = route_for_platform(region)
        url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        return None

    def get_summoner_by_puuid(self, puuid, region='na1'):
        endpoint = "SUMMONERV4"
        route = route_for_platform(region)
        url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        return None

    def get_league_entries_by_summoner(self, summoner_id, region='na1'):
        endpoint = "LEAGUEV4"
        route = route_for_platform(region)
        url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        return None

    def _tier_order(self, tier):
        order = {
            "IRON": 1,
            "BRONZE": 2,
            "SILVER": 3,
            "GOLD": 4,
            "PLATINUM": 5,
            "EMERALD": 6,
            "DIAMOND": 7,
            "MASTER": 8,
            "GRANDMASTER": 9,
            "CHALLENGER": 10,
        }
        return order.get((tier or "").upper(), 0)

#if __name__ == "__main__":
