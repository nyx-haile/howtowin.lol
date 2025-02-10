what is howtowin.lol?

A lol stat tracker that gives you more useful information than 'build this item'

LOL is a situational game. Counters exist, and the game doesn't always go perfectly from level 1. Building blade of the ruined king into thornmail is kinda trolling, but our stat trackers don't reflect this. howtowin.lol rectifies this by normalising all stats over various diffs, (potentially) including gold, team gold, level, team level, experience, and others. 

However, to compute these stats for every single game, every single patch would put my electricity bill through the roof, so I've leveraged my cryptographic skills to build a verified computation algorithm that allows you to access the site by doing some of the number crunching. Hopefully this will allow everyone to get the stats they want and leave us free of ads. 

What does howtowin.lol do?:
Recommends 'best losing item' and 'best winning item'
Recommends 'best item for AP team' and 'Best item for AD team'
Recommends 'best items for $user on $champion'
Recommends 'best skill order' for all of the above
Recommends 'dragon value, herald value, voidgrubs value' as a percentage of winrate.
Detects effective losses:
	effective gold loss - caused by late item buys
	effective XP loss - caused by excessive roaming or dying (show correlation between time out of lane and winrate)
Recommends 'best champion for $user' based on champion winrate with user's expected stats
	calculate user stats,
	normalise for champion,
	calculate champion winrate,
	normalise for stats,
	display user expected winrate on champion


How?
the stats are not super complicated.
All normalisation is done by rescaling everything to have mean 0 and range +/- 1. 
Dimensionality reduction (for the purposes of identifying weakpoints) is done by (decide algorithm)

More research to be done in that area.
