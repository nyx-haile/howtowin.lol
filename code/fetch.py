#Helper functions to fetch data from the Riot Games API
import os
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

# Live-patch gate (handle_player skips stale puuids whose newest match is off-patch)
LIVE_PATCH = os.environ.get("HOWL_LIVE_PATCH", "").strip() or None


def _patch_of(game_version: str) -> str:
    if not game_version:
        return ""
    parts = game_version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else game_version

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

    def handle_player(self, match_count=None, skip_rank=False, route=None):
        if self.player == None:
            self.get_player()
        if match_count is None:
            match_count = int(self.kwargs.get('match_count', 10))

        player_route = route or (self.hget("player_region", self.player) or b"americas").decode()

        # Only fetch match list — skip account/rank lookups to save API calls.
        matches = self.get_matches_by_puuid(self.player, count=match_count, route=player_route)
        self.set("log", f"Player {self.player} has {len(matches)} matches")

        # Stale-player skip: peek at the newest match's gameVersion. If off-patch,
        # don't queue any of this player's matches. Saves up to match_count-1
        # downstream match-v5 + timeline calls per stale puuid. 1 extra match-v5
        # here, but it's amortised against the (often 100) it replaces.
        if LIVE_PATCH and matches:
            try:
                head = self.get_match_by_id(matches[0])
                head_patch = _patch_of((head or {}).get("info", {}).get("gameVersion", ""))
                if head_patch != LIVE_PATCH:
                    self.srem("player_processing", self.player)
                    self.sadd("player_handled", self.player)
                    self.sadd("player_skipped_stale", self.player)
                    return
            except AssertionError:
                # transient API error — fall through and let normal flow handle it
                pass

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
            # league-v4/entries/by-puuid is the modern direct path; summoner-v4 no
            # longer returns `id`. Returns None on 400 decryption mismatch — but
            # during seed crawl we assume match's platform == puuid's home, so
            # single attempt is fine here (backfill does platform fallback).
            entries = self.get_league_entries_by_puuid(self.player, region=platform)
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
            self.zincrby(f"match_queue:{player_route}", 1, *new_matches)
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
        """Block until the per-route backoff clears, then fire the request."""
        kwargs.pop("endpoint", None)
        route = kwargs.pop("route", "americas")
        key = f"ratelimit:{route}:backoff_until"
        while True:
            until = float(self.get(key) or 0)
            wait = until - time.time()
            if wait <= 0:
                return func(*args, **kwargs)
            time.sleep(min(wait, 1.0))

    def request(self, url, headers, endpoint, route="americas"):
        for attempt in range(9):
            response = self.ratelimit(requests.get, url, headers=headers,
                                      endpoint=endpoint, route=route)
            if response.status_code == 200:
                self.incr(f"stats:{route}:requests")
                return response
            if response.status_code != 429:
                return response

            retry_after = float(response.headers.get("Retry-After") or 1)
            wait = retry_after + random.uniform(0.1, 0.5)
            # Broadcast backoff to all workers on this route.
            self.set(f"ratelimit:{route}:backoff_until", time.time() + wait,
                     ex=int(wait) + 30)
            print(f"[429] {route} backing off {wait:.1f}s (attempt {attempt+1})", flush=True)
            time.sleep(wait)
        return response

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

    def get_master_league(self, queue='RANKED_SOLO_5x5', region='na1'):
        endpoint = "LEAGUEV4"
        route = route_for_platform(region)
        url = f"https://{region}.api.riotgames.com/lol/league/v4/masterleagues/by-queue/{queue}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json()
        return None

    def get_league_exp_entries(self, tier, division, queue='RANKED_SOLO_5x5',
                               page=1, region='na1'):
        """league-exp-v4 paginated entries for tiers IRON..DIAMOND.
        Returns a list of entry dicts (with `puuid`) — NOT a league doc.
        """
        endpoint = "LEAGUEEXPV4"
        route = route_for_platform(region)
        url = (f"https://{region}.api.riotgames.com/lol/league-exp/v4/entries/"
               f"{queue}/{tier}/{division}?page={page}")
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json() or []
        return []

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

    def get_league_entries_by_puuid(self, puuid, region='na1'):
        """league-v4/entries/by-puuid — returns league entries directly.
        Returns 400 "Exception decrypting" if puuid's home platform differs.
        Caller should iterate platforms on decryption failure.
        Raises on 429/5xx so caller can distinguish transient failure from
        unranked (vs silently returning []).
        """
        endpoint = "LEAGUEV4"
        route = route_for_platform(region)
        url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
        headers = {"X-Riot-Token": self.api_key}
        response = self.request(url, headers=headers, endpoint=endpoint, route=route)
        if response.status_code == 200:
            return response.json() or []
        if response.status_code == 400:
            # decryption error = wrong region; signal for caller to retry
            return None
        if response.status_code == 404:
            return []
        # 429 (exhausted retries) or 5xx — signal transient failure
        raise RuntimeError(f"league-v4 transient {response.status_code} on {region}")

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
