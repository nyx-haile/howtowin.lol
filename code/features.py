"""Feature computation layer.

After a match is parsed and stored in the frames table, call compute_match()
to derive per-minute stats, team-relative stats, and store them in frame_normalized.

Role mapping from Riot API teamPosition values:
  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY -> normalized for matching opponents.
"""

import sqlite3
import db

_ROLE_NORM = {
    'TOP': 'TOP',
    'JUNGLE': 'JGL',
    'MIDDLE': 'MID',
    'BOTTOM': 'BOT',
    'UTILITY': 'SUP',
}


def _norm_role(role):
    return _ROLE_NORM.get(role, role)


def compute_match(match_id):
    """Compute derived stats for a processed match and write to frame_normalized."""
    conn = db.get_conn()

    game = conn.execute(
        "SELECT winning_team FROM games WHERE match_id = ?", (match_id,)
    ).fetchone()
    if game is None:
        conn.close()
        raise ValueError(f"Match {match_id} not found in games table")

    winning_team = game['winning_team']

    frames = conn.execute(
        """SELECT match_id, participant_slot, puuid, team_id, role,
                  timestamp_ms, current_gold, total_gold, xp, level, cs,
                  kills, deaths, assists, vision_score
           FROM frames WHERE match_id = ?
           ORDER BY timestamp_ms, participant_slot""",
        (match_id,)
    ).fetchall()
    conn.close()

    if not frames:
        return

    # Group by timestamp
    by_ts = {}
    for row in frames:
        ts = row['timestamp_ms']
        if ts not in by_ts:
            by_ts[ts] = []
        by_ts[ts].append(dict(row))

    normalized_rows = []
    for ts, players in sorted(by_ts.items()):
        minute = ts / 60000.0

        # Build lookup: (team_id, normalized_role) -> player data
        by_team_role = {}
        for p in players:
            key = (p['team_id'], _norm_role(p['role']))
            by_team_role[key] = p

        # Compute team totals
        team_gold = {}
        team_level = {}
        for p in players:
            tid = p['team_id']
            team_gold[tid] = team_gold.get(tid, 0) + p['total_gold']
            team_level[tid] = team_level.get(tid, 0) + p['level']

        for p in players:
            slot = p['participant_slot']
            tid = p['team_id']
            role = _norm_role(p['role'])

            # Find opponent (other team, same role)
            enemy_teams = [k for k in team_gold if k != tid]
            opponent = by_team_role.get((enemy_teams[0], role)) if enemy_teams else None

            gold_diff = (p['total_gold'] - opponent['total_gold']) if opponent else 0.0
            xp_diff = (p['xp'] - opponent['xp']) if opponent else 0.0
            cs_diff = (p['cs'] - opponent['cs']) if opponent else 0.0

            enemy_team_id = enemy_teams[0] if enemy_teams else None
            team_gold_lead = (
                team_gold.get(tid, 0) - team_gold.get(enemy_team_id, 0)
                if enemy_team_id else 0.0
            )
            team_level_lead = (
                team_level.get(tid, 0) - team_level.get(enemy_team_id, 0)
                if enemy_team_id else 0.0
            )

            safe_minute = max(minute, 1.0)
            normalized_rows.append({
                'match_id': match_id,
                'participant_slot': slot,
                'puuid': p['puuid'],
                'team_id': tid,
                'role': role,
                'timestamp_ms': ts,
                'minute': minute,
                'cs_per_min': p['cs'] / safe_minute,
                'gold_per_min': p['total_gold'] / safe_minute,
                'xp_per_min': p['xp'] / safe_minute,
                'gold_diff_vs_opponent': float(gold_diff),
                'xp_diff_vs_opponent': float(xp_diff),
                'cs_diff_vs_opponent': float(cs_diff),
                'team_gold_lead': float(team_gold_lead),
                'team_level_lead': float(team_level_lead),
                'current_gold': p['current_gold'],
                'total_gold': p['total_gold'],
                'xp': p['xp'],
                'level': p['level'],
                'cs': p['cs'],
                'kills': p['kills'],
                'deaths': p['deaths'],
                'assists': p['assists'],
                'vision_score': p['vision_score'],
                'win': 1 if tid == winning_team else 0,
            })

    db.insert_frame_normalized(normalized_rows)
    return len(normalized_rows)
