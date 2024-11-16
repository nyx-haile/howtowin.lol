from fetch import agent
from parser import parser
import sys
flag = sys.argv[1]
if flag == "a":
    seed = agent.connect()
    account = seed.get_account_by_riot_id("chaos", "kotic")
    seed.zadd("player_queue", {account["puuid"]: 0}) 
elif flag == "p":
    seed = parser.connect()
    #seed.client.execute("CREATE  IF NOT EXISTS ranked")
    #seed.client.execute("USE ranked")
    #create match vectors here 
    seed.client.execute("CREATE TABLE IF NOT EXISTS test_matches (match_id VARCHAR(255) PRIMARY KEY, players Array(Array(Array(UInt32)))) ENGINE=MergeTree()")
