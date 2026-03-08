#!/usr/local/env python3
from fetch import agent
import json
from dragon import Dragon
import db


class parser(agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dragon = Dragon()

    def handle_match(self):
        if db.game_exists(self.match):
            self.log(f"Match {self.match} already processed, skipping")
            return

        self.match_data = self.get_match_by_id(self.match)
        self.match_timeline = self.get_timeline_by_match(self.match)

        info = self.match_data['info']
        participants = info['participants']

        # Initialize per-participant state
        self.db = {}
        self.puuids = {}
        self.team_ids = {}
        self.roles = {}
        self.winning_team = None

        for p in participants:
            slot = p['participantId']
            self.db[slot] = {
                'kills': 0,
                'deaths': 0,
                'assists': 0,
                'wards_placed': 0,
                'ward_kills': 0,
                'ITEM_STATE': {},
            }
            self.puuids[slot] = p['puuid']
            self.team_ids[slot] = p['teamId']
            self.roles[slot] = p.get('teamPosition', 'UNKNOWN')

        # Process frames — events first so cumulative stats are updated before pframe row is written
        frame_rows = []
        for frame in self.match_timeline['info']['frames']:
            timestamp_ms = frame['timestamp']
            self.handle_events(frame['events'])
            frame_rows.extend(self.handle_pframe(frame['participantFrames'], timestamp_ms))

        # Derive patch string (e.g. "14.3" from "14.3.445.7843")
        version_parts = info.get('gameVersion', '').split('.')
        patch = '.'.join(version_parts[:2]) if len(version_parts) >= 2 else info.get('gameVersion', '')

        db.insert_game(
            match_id=self.match,
            patch=patch,
            queue_id=info.get('queueId', 0),
            game_duration_s=info.get('gameDuration', 0),
            winning_team=self.winning_team or 0,
            created_at=info.get('gameCreation', 0) // 1000,
        )
        db.insert_frames(frame_rows)
        db.insert_players([
            {'puuid': self.puuids[slot]}
            for slot in self.puuids
        ])

        self.log(f"Stored match {self.match}: {len(frame_rows)} frame rows")

    def handle_pframe(self, pframe, timestamp_ms):
        rows = []
        for player_key, pdata in pframe.items():
            slot = int(player_key)
            player_db = self.db[slot]

            cs = pdata.get('minionsKilled', 0) + pdata.get('jungleMinionsKilled', 0)
            pos = pdata.get('position', {})
            vision_score = player_db['wards_placed'] + player_db['ward_kills']

            # Serialize active items (item_id -> count > 0) as a JSON array, up to 7 slots
            items = []
            for item_id, count in player_db['ITEM_STATE'].items():
                if item_id > 0 and count > 0:
                    items.extend([item_id] * count)
            item_ids = json.dumps(items[:7])

            rows.append({
                'match_id': self.match,
                'participant_slot': slot,
                'puuid': self.puuids[slot],
                'team_id': self.team_ids[slot],
                'role': self.roles[slot],
                'timestamp_ms': timestamp_ms,
                'current_gold': pdata.get('currentGold', 0),
                'total_gold': pdata.get('totalGold', 0),
                'xp': pdata.get('xp', 0),
                'level': pdata.get('level', 0),
                'cs': cs,
                'pos_x': pos.get('x', 0),
                'pos_y': pos.get('y', 0),
                'kills': player_db['kills'],
                'deaths': player_db['deaths'],
                'assists': player_db['assists'],
                'vision_score': vision_score,
                'item_ids': item_ids,
            })
        return rows

    def handle_events(self, events):
        for event in events:
            self.handle_event(event)

    def handle_event(self, event):
        match event['type']:
            case 'CHAMPION_KILL':
                killer_id = event.get('killerId', 0)
                victim_id = event.get('victimId', 0)
                assisters = event.get('assistingParticipantIds', [])
                if killer_id > 0:
                    self.db[killer_id]['kills'] += 1
                if victim_id > 0:
                    self.db[victim_id]['deaths'] += 1
                for a in assisters:
                    self.db[a]['assists'] += 1
                return None
            case 'WARD_PLACED':
                creator_id = event.get('creatorId', 0)
                if creator_id > 0:
                    self.db[creator_id]['wards_placed'] += 1
                return None
            case 'WARD_KILL':
                killer_id = event.get('killerId', 0)
                if killer_id > 0:
                    self.db[killer_id]['ward_kills'] += 1
                return None
            case 'ITEM_PURCHASED':
                pid = event.get('participantId', 0)
                if pid > 0:
                    item_id = event['itemId']
                    self.db[pid]['ITEM_STATE'][item_id] = self.db[pid]['ITEM_STATE'].get(item_id, 0) + 1
                return None
            case 'ITEM_SOLD' | 'ITEM_DESTROYED':
                pid = event.get('participantId', 0)
                if pid > 0:
                    item_id = event['itemId']
                    count = self.db[pid]['ITEM_STATE'].get(item_id, 0)
                    if count > 1:
                        self.db[pid]['ITEM_STATE'][item_id] = count - 1
                    elif count == 1:
                        del self.db[pid]['ITEM_STATE'][item_id]
                return None
            case 'ITEM_UNDO':
                pid = event.get('participantId', 0)
                if pid > 0:
                    before_id = event.get('beforeId', 0)
                    after_id = event.get('afterId', 0)
                    if before_id > 0:
                        count = self.db[pid]['ITEM_STATE'].get(before_id, 0)
                        if count > 1:
                            self.db[pid]['ITEM_STATE'][before_id] = count - 1
                        elif count == 1:
                            del self.db[pid]['ITEM_STATE'][before_id]
                    if after_id > 0:
                        self.db[pid]['ITEM_STATE'][after_id] = self.db[pid]['ITEM_STATE'].get(after_id, 0) + 1
                return None
            case 'GAME_END':
                self.winning_team = event.get('winningTeam', 0)
                return None
            case 'BUILDING_KILL' | 'TURRET_PLATE_DESTROYED' | 'ELITE_MONSTER_KILL':
                return None
            case 'PAUSE_END' | 'LEVEL_UP' | 'SKILL_LEVEL_UP' | 'CHAMPION_SPECIAL_KILL':
                return None
            case 'OBJECTIVE_BOUNTY_PRESTART' | 'OBJECTIVE_BOUNTY_FINISH':
                return None
        self.log(f"Unknown event: {event['type']}")


if __name__ == "__main__":
    match = parser.connect()
    match.match = "NA1_5081314144"
    match.handle_match()
