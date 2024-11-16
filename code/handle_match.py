"""
def handle_match(match):
    #get the players
    players = match["info"]["participants"]
    for player in players:
	    #check if the player is already in the database
    	#convert to redis
        player_cursor.execute("SELECT * FROM players WHERE puuid=?", (player["puuid"],))
        if player_cursor.fetchone() == None:
	        #add the player to the database and store the timestamp of their matches
            #may be used later for a prediction algorithm to make checking more efficient.
	        player_cursor.execute("INSERT INTO players (puuid) VALUES (?)", (player["puuid"],))
	        player_db.commit()
	        #get the matches of the player
            matches = get_matches_by_puuid(player["puuid"])
            if matches != None:
                for m in matches:
                    handle_match(m)
            else:
		        print("Failed to get matches for player " + player["puuid"])
        else:
	        print("Player " + player["puuid"] + " is already in the database")
"""
