from fetch import agent

seed = agent.connect()
account = seed.get_account_by_riot_id("chaos", "kotic")
seed.zadd("player_queue", {account["puuid"]: 0}) 

