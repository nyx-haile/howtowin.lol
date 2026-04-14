import json
from fetch import agent
from db import get_conn, init_db, insert_game, insert_frame, insert_player, insert_event

ROLE_MAP = {
    'TOP': 'TOP', 'JUNGLE': 'JGL', 'MIDDLE': 'MID',
    'BOTTOM': 'BOT', 'UTILITY': 'SUP', '': 'UNK'
}


class parser(agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_db()

    def handle_match(self):
        self.match_data = self.get_match_by_id(self.match)
        self.match_timeline = self.get_timeline_by_match(self.match)

        match_id = self.match_data['metadata']['matchId']
        info = self.match_data['info']

        # Build participant mapping
        self.participants = {}
        winning_team = None
        for p in info['participants']:
            slot = p['participantId']
            self.participants[slot] = {
                'puuid': p['puuid'],
                'team_id': p['teamId'],
                'role': ROLE_MAP.get(p.get('teamPosition', ''), 'UNK'),
            }
            if p.get('win'):
                winning_team = p['teamId']

        # Cumulative event counters per participant
        self.cumulative = {}
        for slot in range(1, 11):
            self.cumulative[slot] = {
                'kills': 0, 'deaths': 0, 'assists': 0,
                'wards_placed': 0, 'wards_killed': 0, 'items': []
            }

        conn = get_conn()

        # Insert game
        insert_game(conn, match_id, info.get('gameVersion', ''),
                     info.get('queueId', 0), info.get('gameDuration', 0),
                     winning_team or 0, info.get('gameCreation', 0))

        # Insert players
        for slot, pdata in self.participants.items():
            riot_id = None
            for mp in info['participants']:
                if mp['puuid'] == pdata['puuid']:
                    name = mp.get('riotIdGameName', '')
                    tag = mp.get('riotIdTagline', '')
                    if name:
                        riot_id = f"{name}#{tag}"
                    break
            insert_player(conn, pdata['puuid'], riot_id)

        # Process timeline
        for frame in self.match_timeline['info']['frames']:
            ts_ms = frame['timestamp']
            for event in frame['events']:
                self._store_event(conn, match_id, event)
                self._accumulate(event)
            self._store_frames(conn, match_id, ts_ms, frame['participantFrames'])

        conn.commit()
        conn.close()
        self.log(f"Parsed {match_id}")

    def _store_event(self, conn, match_id, event):
        """Store every event as a row — the feature layer mines these."""
        etype = event['type']
        pos = event.get('position', {})
        details = {k: v for k, v in event.items()
                   if k not in ('type', 'timestamp', 'position',
                                'participantId', 'killerId', 'victimId')}

        insert_event(
            conn, match_id, event.get('timestamp', 0), etype,
            participant_id=event.get('participantId'),
            killer_id=event.get('killerId'),
            victim_id=event.get('victimId'),
            killer_team=event.get('killerTeamId'),
            team_id=event.get('teamId'),
            position_x=pos.get('x'),
            position_y=pos.get('y'),
            details=json.dumps(details)
        )

    def _accumulate(self, event):
        """Update per-player cumulative counters from events."""
        etype = event['type']

        if etype == 'CHAMPION_KILL':
            k = event.get('killerId', 0)
            v = event.get('victimId', 0)
            if 1 <= k <= 10:
                self.cumulative[k]['kills'] += 1
            if 1 <= v <= 10:
                self.cumulative[v]['deaths'] += 1
            for a in event.get('assistingParticipantIds', []):
                if 1 <= a <= 10:
                    self.cumulative[a]['assists'] += 1

        elif etype == 'WARD_PLACED':
            c = event.get('creatorId', 0)
            if 1 <= c <= 10:
                self.cumulative[c]['wards_placed'] += 1

        elif etype == 'WARD_KILL':
            k = event.get('killerId', 0)
            if 1 <= k <= 10:
                self.cumulative[k]['wards_killed'] += 1

        elif etype == 'ITEM_PURCHASED':
            p = event.get('participantId', 0)
            if 1 <= p <= 10:
                self.cumulative[p]['items'].append(event['itemId'])

        elif etype == 'ITEM_SOLD' or etype == 'ITEM_DESTROYED':
            p = event.get('participantId', 0)
            iid = event.get('itemId', 0)
            if 1 <= p <= 10 and iid in self.cumulative[p]['items']:
                self.cumulative[p]['items'].remove(iid)

        elif etype == 'ITEM_UNDO':
            p = event.get('participantId', 0)
            before = event.get('beforeId', 0)
            after = event.get('afterId', 0)
            if 1 <= p <= 10:
                if before and before in self.cumulative[p]['items']:
                    self.cumulative[p]['items'].remove(before)
                if after:
                    self.cumulative[p]['items'].append(after)

    def _store_frames(self, conn, match_id, ts_ms, participant_frames):
        """Write one frames row per player at this timestamp."""
        for slot_str, pf in participant_frames.items():
            slot = int(slot_str)
            if slot not in self.participants:
                continue
            pdata = self.participants[slot]
            cum = self.cumulative[slot]
            pos = pf.get('position', {})

            insert_frame(
                conn, match_id, slot,
                pdata['puuid'], pdata['team_id'], pdata['role'], ts_ms,
                current_gold=pf.get('currentGold', 0),
                total_gold=pf.get('totalGold', 0),
                xp=pf.get('xp', 0),
                level=pf.get('level', 1),
                cs=pf.get('minionsKilled', 0),
                jungle_cs=pf.get('jungleMinionsKilled', 0),
                pos_x=pos.get('x', 0),
                pos_y=pos.get('y', 0),
                kills=cum['kills'],
                deaths=cum['deaths'],
                assists=cum['assists'],
                ward_count=cum['wards_placed'] + cum['wards_killed'],
                item_ids=list(cum['items'])
            )


if __name__ == "__main__":
    p = parser.connect()
    p.match = "NA1_5081314144"
    p.handle_match()
    print("Done")
