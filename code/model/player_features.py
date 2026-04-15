"""Per-puuid crafted feature vector.

Fields (total 16):
  0: rank_tier_ordinal     (0 for unranked; higher = higher tier)
  1: lp_normalized         (lp / 1000, capped at 2.0 for Masters+)
  2: games_in_corpus       (log1p)
  3: winrate_in_corpus     (wins / games, 0 if games=0)
  4: avg_kda               ((kills+assists)/max(deaths,1))
  5: avg_cs_at_10          (total CS at minute 10, averaged across corpus)
  6: avg_gold_at_10
  7: avg_damage_to_champs
  8: avg_wards_placed
  9: avg_wards_killed
 10: main_role_top
 11: main_role_jungle
 12: main_role_mid
 13: main_role_bot
 14: main_role_support
 15: reserved
"""
import numpy as np

from db import get_conn

PLAYER_FEATURE_DIM = 16

RANK_TIER_ORDER = {
    "IRON": 1, "BRONZE": 2, "SILVER": 3, "GOLD": 4, "PLATINUM": 5,
    "EMERALD": 6, "DIAMOND": 7, "MASTER": 8, "GRANDMASTER": 9, "CHALLENGER": 10,
}

ROLE_INDEX = {"TOP": 10, "JGL": 11, "MID": 12, "BOT": 13, "SUP": 14}


def player_feature_vector(puuid):
    vec = np.zeros(PLAYER_FEATURE_DIM, dtype=np.float32)
    conn = get_conn()
    try:
        prow = conn.execute(
            "SELECT rank_tier, lp FROM players WHERE puuid = ?", (puuid,)
        ).fetchone()
        if prow is None:
            return vec

        if prow["rank_tier"]:
            vec[0] = float(RANK_TIER_ORDER.get(prow["rank_tier"], 0))
        if prow["lp"] is not None:
            vec[1] = min(float(prow["lp"]) / 1000.0, 2.0)

        # Games in corpus for this puuid.
        games = conn.execute(
            """SELECT f.match_id, f.team_id, g.winning_team
               FROM frames f JOIN games g USING(match_id)
               WHERE f.puuid = ? GROUP BY f.match_id""",
            (puuid,)
        ).fetchall()
        n = len(games)
        vec[2] = float(np.log1p(n))
        if n > 0:
            wins = sum(1 for g in games if g["team_id"] == g["winning_team"])
            vec[3] = float(wins) / float(n)

        # Per-game aggregates at minute 10 (timestamp ~600000ms).
        rows = conn.execute(
            """SELECT cs, total_gold, kills, deaths, assists, ward_count
               FROM frames WHERE puuid = ? AND timestamp_ms BETWEEN 540000 AND 660000""",
            (puuid,)
        ).fetchall()
        if rows:
            cs = [r["cs"] for r in rows]
            gold = [r["total_gold"] for r in rows]
            kda = [(r["kills"] + r["assists"]) / max(r["deaths"], 1) for r in rows]
            vec[4] = float(np.mean(kda))
            vec[5] = float(np.mean(cs))
            vec[6] = float(np.mean(gold))
            vec[8] = float(np.mean([r["ward_count"] for r in rows]))

        # Damage placeholder (not populated in frames table; leave 0).
        # Damage will be added when the schema carries it; zero is a safe default.

        # Wards killed approximated by ward_count delta; skipping for Plan A.

        # Role frequency.
        roles = conn.execute(
            "SELECT role, COUNT(*) c FROM frames WHERE puuid = ? GROUP BY role",
            (puuid,)
        ).fetchall()
        total = sum(r["c"] for r in roles) or 1
        for r in roles:
            idx = ROLE_INDEX.get(r["role"])
            if idx is not None:
                vec[idx] = float(r["c"]) / float(total)
    finally:
        conn.close()
    return vec
