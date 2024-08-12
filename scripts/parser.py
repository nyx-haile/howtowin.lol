#!/usr/local/env python3
from fetch import agent
import json
#import cstat
from dragon import Dragon
import requests
from clickhouse_driver import Client
from collections.abc import MutableMapping

def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

class sparse_list(list):
    def __init__(self, *args, **kwargs):
        self.vals = kwargs
        self.empty = None
        self.maxindex = -1
    def __getitem__(self, index):
        if index < self.maxindex:
            return self.vals.get(index, self.empty)
        else:
            raise IndexError
    def __setitem__(self, index, value):
        self.vals[index] = value
        if index > self.maxindex:
            self.maxindex = index
    def __delitem__(self, index):
        self.vals.pop(index)
        if index == self.maxindex:
            self.maxindex = max(self.vals.keys())
    def __str__(self):
        buf = []
        for key in self.vals:
            if len(buf) == key:
                buf.append(self.vals[key])
            else:
                buf.extend([self.empty] * (key - len(buf)))
                buf.append(self.vals[key])
        for key in self.vals:
            assert buf[key] == self.vals[key], "You fucked up somehow..."
        return str(buf)
    def __len__(self):
        return self.maxindex +1


class parser(agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dragon = Dragon()
        with open("../secrets/clickhouse", "r") as f:
            self.clickhouse_key = f.read().strip()
        self.client = Client(user="howl", password=f"{self.clickhouse_key}", host="howtowin-lol-presales-test.c.aivencloud.com", port=21168, secure=True)

    #def get_match(self):
        #self.match = "NA1_5007766029"
        #super.get_match() 
        #return self.match
        #idk if this will work actually...
        #maybe i should rank the matches by elo?
        #or maybe i should just get the most recent match?
        #lol ai 
        #i think i should just get the most recent match
        #ok.
    def handle_match(self):
        self.match_data = self.get_match_by_id(self.match)
        self.match_timeline = self.get_timeline_by_match(self.match)
        self.match_SL= sparse_list()
        for frame in self.match_timeline['info']['frames']:
            self.timestamp = frame['timestamp']
            self.handle_pframe(frame['participantFrames'])
            self.handle_events(frame['events'])
        self.client.execute(f"INSERT INTO test_matches {self.match_SL}")
    def handle_pframe(self, pframe):
        parray = []
        for player in pframe:
            nframe = flatten(pframe[player])
            nkeys = list(nframe.keys())
            nkeys.sort()
            parray.append([nframe[key] for key in nkeys])
        self.match_SL[self.timestamp] = parray
        print(self.timestamp)
        print(len(self.match_SL))
        self.match_SL[10] = "test"
        print(f"{self.match_SL}")
        assert False
    def handle_events(self, events):
        for event in events:
            self.handle_event(event)
        
    def arrayify(self, some_dict):
        pass

    def handle_event(self, event):
        pass
    def mangle_event(self, event):
        match event['type']:
            case "ITEM_UNDO":
                db[event['participantId']]['ITEMS'][event['timestamp']] = (-event['beforeId'], event['afterId'])
                db[event['participantId']]['ITEM_STATE'][event['beforeId']] -=1
                db[event['participantId']]['ITEM_STATE'][event['afterId']] +=1
                db[event['participantId']]['ITEMS']['SET'+str(event['timestamp'])] = json.dumps(db[event['participantId']]['ITEM_STATE'])
                
                #db[event['participantId']]['GOLD']['timestamp'] =  #update gold totals
                #dict_keys(['afterId', 'beforeId', 'goldGain', 'participantId', 'timestamp', 'type'])
                return None
            case 'BUILDING_KILL':
                #update gold totals
                #dict_keys(['assistingParticipantIds', 'bounty', 'buildingType', 'killerId', 'laneType', 'position', 'teamId', 'timestamp', 'towerType', 'type']),
                return None
            case  'WARD_PLACED':
                #WARD_PLACED': dict_keys(['creatorId', 'timestamp', 'type', 'wardType']),
                return None
            case  'TURRET_PLATE_DESTROYED':
                #dict_keys(['killerId', 'laneType', 'position', 'teamId', 'timestamp', 'type']),
                return None
            case  'PAUSE_END':
                #pass
                return None
            case 'LEVEL_UP':
                #dict_keys(['level', 'participantId', 'timestamp', 'type'])
                return None
            case 'CHAMPION_SPECIAL_KILL':
                #dict_keys(['killType', 'killerId', 'position', 'timestamp', 'type']),
                return None
            case  'GAME_END':
                with open(f'../DB/{matchID}_info', 'a') as match_row:
                    match_row.write(str(event['winningTeam'])+'\n')
                    match_row.write(str(event['realTimestamp']))
                #(['gameId', 'realTimestamp', 'timestamp', 'type', 'winningTeam'])}
                return None
            case 'OBJECTIVE_BOUNTY_PRESTART':
                #dict_keys(['actualStartTime', 'teamId', 'timestamp', 'type']),
                return None
            case  'SKILL_LEVEL_UP':
                #dict_keys(['levelUpType', 'participantId', 'skillSlot', 'timestamp', 'type'])
                return None
            case 'ITEM_DESTROYED':
                db[event['participantId']]['ITEMS'][event['timestamp']] = -event['itemId']
                db[event['participantId']]['ITEM_STATE'].setdefault(event['itemId'], 0)
                db[event['participantId']]['ITEM_STATE'][event['itemId']] -=1
                db[event['participantId']]['ITEMS']['SET'+str(event['timestamp'])] = json.dumps(db[event['participantId']]['ITEM_STATE'])
                #dict_keys(['itemId', 'participantId', 'timestamp', 'type'])
                return None
            case 'ITEM_PURCHASED':
                db[event['participantId']]['ITEMS'][event['timestamp']] = event['itemId']
                db[event['participantId']]['ITEM_STATE'].setdefault(event['itemId'], 0)
                db[event['participantId']]['ITEM_STATE'][event['itemId']] +=1
                db[event['participantId']]['ITEMS']['SET'+str(event['timestamp'])] = json.dumps(db[event['participantId']]['ITEM_STATE'])
                #'itemId', 'participantId', 'timestamp', 'type'
                return None
            case 'CHAMPION_KILL':
                #dict_keys(['assistingParticipantIds', 'bounty', 'killStreakLength', 'killerId', 'position', 'shutdownBounty', 'timestamp', 'type', 'victimDamageDealt', 'victimDamageReceived', 'victimId'])
                return None
            case 'ITEM_SOLD':
                db[event['participantId']]['ITEMS'][event['timestamp']] = -event['itemId']
                db[event['participantId']]['ITEM_STATE'].setdefault(event['itemId'], 0)
                db[event['participantId']]['ITEM_STATE'][event['itemId']] -=1
                db[event['participantId']]['ITEMS']['SET'+str(event['timestamp'])] = json.dumps(db[event['participantId']]['ITEM_STATE'])
                #dict_keys(['itemId', 'participantId', 'timestamp', 'type']),
                return None
            case 'WARD_KILL':
                #dict_keys(['killerId', 'timestamp', 'type', 'wardType'])
                return None
            case 'ELITE_MONSTER_KILL':
                #dict_keys(['bounty', 'killerId', 'killerTeamId', 'monsterSubType', 'monsterType', 'position', 'timestamp', 'type']),
                return None
        assert False, "Unknown Event Type"
    itemtypes={}




if __name__ == "__main__":
    match = parser.connect()
    match.match = "NA1_5081314144"
    match.handle_match()


#    #get matchId
#    matchID = game['metadata']['matchId']
#
#    #get ID assignments
#    participantID = {}
#    for participantDto in game['info']['participants']:
#        participantID[participantDto['participantId']]=participantDto['puuid']
#        #write PUUIDs to database	
#
#    #write db Files
#    db = {}
#    for playerID in participantID.keys():
#        db[playerID] = {}
#        db[playerID]['ITEMS'] = {0: 0}
#        db[playerID]['ITEM_STATE']={0: 0}
#        #db[playerID]['DAMAGE'] = {}
#        #db[playerID]['WARDS'] = {}
#        #db[playerID]['BOUNTY'] = {}
#        #db[playerID]['OBJECTIVES'] = {}
#        #for dbStat in cstat.mvp_stats:
#            #db[playerID][dbStat] = {}
#    #build game row
#    with open('../DB/'+matchID+'_'+'info', 'w') as match_row:
#        match_row.write(json.dumps(game['metadata']['matchId'])+'\n')
#        match_row.write(json.dumps(game['info']['endOfGameResult'])+'\n')
#        match_row.write(json.dumps(participantID)+'\n')
#
#    def test():   #build timeline rows
#        for index, frame in enumerate(game['info']['frames']):	
#            pass
#            #get events
#        for event in frame['events']:
#            #log event
#            #print(event.keys())
#            #print(event['type'])
#            handle_event(event)
#        #for player in #frame['participantFrames']:
#            #player_frame = frame['participantFrames'][player]
#            #print(player_frame)
#            #for dbStat in ['currentGold', 'totalGold', 'xp']:
#                #stat = player_frame[dbStat]
#                #print(str(stat))
#                #db[int(player)][dbStat][frame['timestamp']] = stat		
#
#    for playerID in db.keys():
#        for dbStat in db[playerID].keys():
#            with open(f'../DB/{matchID}_{participantID[playerID]}_{dbStat}', 'w') as buf:
#                json.dump(db[playerID][dbStat], buf)
#
#    #print([f'{item}: {db[1]['ITEM_STATE'][item]}' for item in db[1]['ITEM_STATE'] if db[1]['ITEM_STATE'][item] > 0])
#    #print([db[1]['ITEMS'][item] for item in db[1]['ITEMS'].keys() if type(item) != str])
