import sqlite3

#push a key to the db

match_db = sqlite3.connect('../db/match.db')
match_cursor = match_db.cursor()
player_db = sqlite3.connect('../db/player.db')
player_cursor = player_db.cursor()

#match_cursor.execute("CREATE TABLE matches (id TEXT PRIMARY KEY, timestamp INTEGER, player1 TEXT, player2 TEXT, player3 TEXT, player4 TEXT, player5 TEXT, player6 TEXT, player7 TEXT, player8 TEXT, player9 TEXT, player10 TEXT, winner INTEGER)")
player_cursor.execute("CREATE TABLE players (puuid TEXT PRIMARY KEY, last_match INTEGER, match_id INTEGER, match_result INTEGER)")

player_cursor.execute("INSERT INTO players (puuid) VALUES ('jzHYaBATS33X6EKmKPnIfmMViHErtxjhAq_bHeamH1Ov4Q7C-QfzDZzF45QxHftoDzEQnCFSL6Xycg')")

print(player_cursor.execute("SELECT * FROM players").fetchall())
