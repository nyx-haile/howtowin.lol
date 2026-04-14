import json
from db import get_conn


def compute_game_states(match_id, db_path=None):
    """Compute team-level macro state at each timestamp for a match.

    Returns: {timestamp_ms: {'team_100': {...}, 'team_200': {...}}}
    """
    conn = get_conn(db_path)

    frames = conn.execute(
        "SELECT * FROM frames WHERE match_id = ? ORDER BY timestamp_ms, participant_slot",
        (match_id,)
    ).fetchall()

    events = conn.execute(
        "SELECT * FROM events WHERE match_id = ? ORDER BY timestamp_ms",
        (match_id,)
    ).fetchall()

    game = conn.execute(
        "SELECT * FROM games WHERE match_id = ?", (match_id,)
    ).fetchone()

    conn.close()

    if not frames or not game:
        return {}

    # Pre-compute cumulative event counts up to each timestamp
    objective_counts = _count_objectives(events)
    tower_counts = _count_towers(events)

    # Group frames by timestamp
    by_ts = {}
    for f in frames:
        ts = f['timestamp_ms']
        if ts not in by_ts:
            by_ts[ts] = []
        by_ts[ts].append(dict(f))

    # Build cumulative snapshots at frame timestamps
    # (objective/tower events may happen between frames)
    all_event_ts = sorted(objective_counts.keys() | tower_counts.keys())

    states = {}
    for ts in sorted(by_ts.keys()):
        players = by_ts[ts]
        team_100 = [p for p in players if p['team_id'] == 100]
        team_200 = [p for p in players if p['team_id'] == 200]

        # Find most recent objective/tower snapshot at or before this frame
        obj_snapshot = {100: {}, 200: {}}
        twr_snapshot = {100: 0, 200: 0}
        for ets in all_event_ts:
            if ets > ts:
                break
            if ets in objective_counts:
                obj_snapshot = objective_counts[ets]
            if ets in tower_counts:
                twr_snapshot = tower_counts[ets]

        states[ts] = {
            'team_100': _team_state(team_100, obj_snapshot.get(100, {}), twr_snapshot.get(100, 0)),
            'team_200': _team_state(team_200, obj_snapshot.get(200, {}), twr_snapshot.get(200, 0)),
        }

    return states


def _team_state(players, objectives, tower_kills):
    """Compute macro features for one team at one timestamp."""
    if not players:
        return {}

    total_gold = sum(p['total_gold'] for p in players)
    total_xp = sum(p['xp'] for p in players)
    total_cs = sum(p['cs'] + p['jungle_cs'] for p in players)
    total_wards = sum(p['ward_count'] for p in players)
    total_kills = sum(p['kills'] for p in players)
    total_deaths = sum(p['deaths'] for p in players)

    # Team centroid and spread
    xs = [p['pos_x'] for p in players]
    ys = [p['pos_y'] for p in players]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    spread = (sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys)) / len(xs)) ** 0.5

    return {
        'total_gold': total_gold,
        'total_xp': total_xp,
        'total_cs': total_cs,
        'ward_count': total_wards,
        'kills': total_kills,
        'deaths': total_deaths,
        'dragon_count': objectives.get('DRAGON', 0),
        'baron_count': objectives.get('BARON_NASHOR', 0),
        'herald_count': objectives.get('RIFTHERALD', 0),
        'horde_count': objectives.get('HORDE', 0),
        'tower_count': tower_kills,
        'centroid_x': cx,
        'centroid_y': cy,
        'spread': spread,
        'avg_level': sum(p['level'] for p in players) / len(players),
    }


def _count_objectives(events):
    """Build cumulative objective counts: {timestamp: {team: {type: count}}}."""
    cumulative = {100: {}, 200: {}}
    result = {}

    for e in events:
        if e['event_type'] == 'ELITE_MONSTER_KILL':
            team = e['killer_team']
            if team not in (100, 200):
                continue
            details = json.loads(e['details']) if e['details'] else {}
            monster = details.get('monsterType', 'UNKNOWN')
            cumulative[team][monster] = cumulative[team].get(monster, 0) + 1

        # Snapshot at each event timestamp
        ts = e['timestamp_ms']
        result[ts] = {
            100: dict(cumulative[100]),
            200: dict(cumulative[200]),
        }

    return result


def _count_towers(events):
    """Build cumulative tower kill counts: {timestamp: {killing_team: count}}.
    Note: teamId in BUILDING_KILL is the team that LOST the tower."""
    cumulative = {100: 0, 200: 0}
    result = {}

    for e in events:
        if e['event_type'] == 'BUILDING_KILL':
            details = json.loads(e['details']) if e['details'] else {}
            if details.get('buildingType') == 'TOWER_BUILDING':
                lost_team = e['team_id']
                if lost_team == 100:
                    cumulative[200] += 1
                elif lost_team == 200:
                    cumulative[100] += 1

        ts = e['timestamp_ms']
        result[ts] = dict(cumulative)

    return result


def compute_feature_vectors(match_id, db_path=None):
    """Compute flat feature vectors for classifier training.

    Returns one vector per team per timestamp — the classifier learns
    which team-level features predict wins.
    """
    conn = get_conn(db_path)
    game = conn.execute("SELECT * FROM games WHERE match_id = ?", (match_id,)).fetchone()
    conn.close()

    if not game:
        return []

    states = compute_game_states(match_id, db_path)
    winning_team = game['winning_team']
    vectors = []

    for ts, team_states in states.items():
        t100 = team_states['team_100']
        t200 = team_states['team_200']

        if not t100 or not t200:
            continue

        for team_id, mine, theirs in [(100, t100, t200), (200, t200, t100)]:
            vectors.append({
                'match_id': match_id,
                'timestamp_ms': ts,
                'team_id': team_id,
                'minutes': ts / 60000.0,

                # Differentials
                'team_gold_lead': mine['total_gold'] - theirs['total_gold'],
                'team_xp_lead': mine['total_xp'] - theirs['total_xp'],
                'team_cs_lead': mine['total_cs'] - theirs['total_cs'],
                'team_kill_lead': mine['kills'] - theirs['kills'],
                'team_ward_diff': mine['ward_count'] - theirs['ward_count'],
                'team_dragon_diff': mine['dragon_count'] - theirs['dragon_count'],
                'team_tower_diff': mine['tower_count'] - theirs['tower_count'],
                'team_baron_diff': mine['baron_count'] - theirs['baron_count'],
                'team_herald_diff': mine['herald_count'] - theirs['herald_count'],

                # Absolute state
                'team_total_gold': mine['total_gold'],
                'team_avg_level': mine['avg_level'],
                'team_dragons': mine['dragon_count'],
                'team_towers': mine['tower_count'],
                'team_barons': mine['baron_count'],
                'team_wards': mine['ward_count'],
                'team_spread': mine['spread'],

                # Outcome
                'won': 1 if team_id == winning_team else 0,
            })

    return vectors


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        match_id = sys.argv[1]
        db_path = sys.argv[2] if len(sys.argv) > 2 else None
        vecs = compute_feature_vectors(match_id, db_path)
        print(f"Computed {len(vecs)} feature vectors for {match_id}")
        if vecs:
            print("Feature keys:", sorted(vecs[0].keys()))
    else:
        print("Usage: python features.py <match_id> [db_path]")
