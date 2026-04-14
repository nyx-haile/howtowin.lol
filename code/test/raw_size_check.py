import requests
import json
import gzip
import sys

with open('../../secrets/api_key') as f:
    API_KEY = f.read().strip()

HEADERS = {"X-Riot-Token": API_KEY}
GAME_NAME = sys.argv[1] if len(sys.argv) > 1 else 'chaos'
TAG_LINE = sys.argv[2] if len(sys.argv) > 2 else 'kotic'


def get(url):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()


account = get(f'https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{GAME_NAME}/{TAG_LINE}')
puuid = account['puuid']
print(f'puuid: {puuid}')

matches = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=1')
match_id = matches[0]
print(f'match: {match_id}')

match = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}')
timeline = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline')

match_json = json.dumps(match, separators=(',', ':'))
timeline_json = json.dumps(timeline, separators=(',', ':'))

match_gz = gzip.compress(match_json.encode())
timeline_gz = gzip.compress(timeline_json.encode())

kb = lambda b: f'{len(b) / 1024:.1f} KB'
print()
print(f'match raw:      {kb(match_json)}')
print(f'match gzipped:  {kb(match_gz)}')
print(f'timeline raw:   {kb(timeline_json)}')
print(f'timeline gzip:  {kb(timeline_gz)}')
print(f'total raw:      {kb(match_json + timeline_json)}')
print(f'total gzipped:  {kb(match_gz + timeline_gz)}')
print()
print(f'at 10k matches gzipped: {(len(match_gz) + len(timeline_gz)) * 10000 / 1024 / 1024 / 1024:.2f} GB')
print(f'at 100k matches gzip:   {(len(match_gz) + len(timeline_gz)) * 100000 / 1024 / 1024 / 1024:.2f} GB')
