import sqlite3
import os

match_db = sqlite3.connect('../db/match.db')
match_cursor = match_db.cursor()
player_db = sqlite3.connect('../db/player.db')
player_cursor = player_db.cursor()

match_cursor.execute("CREATE TABLE matches (match_id TEXT PRIMARY KEY, parsed BOOL, imported   BOOL)")
match_cursor.execute("CREATE TABLE match_queue (match_id TEXT PRIMARY KEY)"

player_cursor.execute("CREATE TABLE players (puuid TEXT PRIMARY KEY, last_check INTEGER, last_game_time INTEGER, last_match INTEGER, empty_checks INTEGER)")
player_cursor.execute("CREATE TABLE player_queue (puuid TEXT PRIMARY KEY)")


player_cursor.execute("INSERT INTO players (puuid) VALUES ('jzHYaBATS33X6EKmKPnIfmMViHErtxjhAq_bHeamH1Ov4Q7C-QfzDZzF45QxHftoDzEQnCFSL6Xycg')")
player_cursor.execute("INSERT INTO player_queue (puuid) VALUES ('jzHYaBATS33X6EKmKPnIfmMViHErtxjhAq_bHeamH1Ov4Q7C-QfzDZzF45QxHftoDzEQnCFSL6Xycg')")
player_db.commit()
match_db.commit()
match_db.close()
player_db.close()


