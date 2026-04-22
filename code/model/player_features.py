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
from model.player_match_stats import PLAYER_MATCH_STATS_TABLE, ensure_player_match_stats

PLAYER_FEATURE_DIM = 16

RANK_TIER_ORDER = {
    "IRON": 1, "BRONZE": 2, "SILVER": 3, "GOLD": 4, "PLATINUM": 5,
    "EMERALD": 6, "DIAMOND": 7, "MASTER": 8, "GRANDMASTER": 9, "CHALLENGER": 10,
}

ROLE_INDEX = {"TOP": 10, "JGL": 11, "MID": 12, "BOT": 13, "SUP": 14}


def _causal_cutoff(conn, puuid, exclude_match_ids):
    """Return the earliest created_at among excluded matches for this puuid.

    Every game at or after this timestamp is excluded (causal cutoff: the model
    must not see *any* of the player's future once one held-out game is hit).
    Returns None if no excluded match belongs to this puuid, meaning no
    filtering is needed.
    """
    if not exclude_match_ids:
        return None
    placeholders = ",".join("?" for _ in exclude_match_ids)
    row = conn.execute(
        f"""SELECT MIN(created_at) AS cutoff
            FROM {PLAYER_MATCH_STATS_TABLE}
            WHERE puuid = ? AND match_id IN ({placeholders})
            GROUP BY puuid""",
        (puuid, *exclude_match_ids),
    ).fetchone()
    if row is None:
        return None
    return row["cutoff"]


def player_feature_vector(puuid, exclude_match_ids=None):
    vec = np.zeros(PLAYER_FEATURE_DIM, dtype=np.float32)
    conn = get_conn()
    try:
        ensure_player_match_stats(conn)
        prow = conn.execute(
            "SELECT rank_tier, lp FROM players WHERE puuid = ?", (puuid,)
        ).fetchone()
        if prow is None:
            return vec

        if prow["rank_tier"]:
            vec[0] = float(RANK_TIER_ORDER.get(prow["rank_tier"], 0))
        if prow["lp"] is not None:
            vec[1] = min(float(prow["lp"]) / 1000.0, 2.0)

        # Determine causal cutoff timestamp for this puuid.
        cutoff = _causal_cutoff(conn, puuid, exclude_match_ids)

        where_clause = f"puuid = ?"
        params = [puuid]
        if cutoff is not None:
            where_clause += " AND created_at < ?"
            params.append(cutoff)

        agg = conn.execute(
            f"""SELECT
                    COUNT(*) AS games_in_corpus,
                    SUM(CASE WHEN team_id = winning_team THEN 1 ELSE 0 END) AS wins_in_corpus,
                    SUM(window_10_rows) AS window_rows,
                    SUM(kda_10_sum) AS kda_10_sum,
                    SUM(cs_10_sum) AS cs_10_sum,
                    SUM(total_gold_10_sum) AS total_gold_10_sum,
                    SUM(ward_count_10_sum) AS ward_count_10_sum
                FROM {PLAYER_MATCH_STATS_TABLE}
                WHERE {where_clause}""",
            params,
        ).fetchone()

        n = int(agg["games_in_corpus"] or 0)
        vec[2] = float(np.log1p(n))
        if n > 0:
            wins = float(agg["wins_in_corpus"] or 0)
            vec[3] = wins / float(n)

        window_rows = float(agg["window_rows"] or 0.0)
        if window_rows > 0:
            vec[4] = float(agg["kda_10_sum"] or 0.0) / window_rows
            vec[5] = float(agg["cs_10_sum"] or 0.0) / window_rows
            vec[6] = float(agg["total_gold_10_sum"] or 0.0) / window_rows
            vec[8] = float(agg["ward_count_10_sum"] or 0.0) / window_rows

        # Damage placeholder (not populated in frames table; leave 0).
        # Damage will be added when the schema carries it; zero is a safe default.

        # Wards killed approximated by ward_count delta; skipping for Plan A.

        # Role frequency.
        roles = conn.execute(
            f"""SELECT role, COUNT(*) c
                FROM {PLAYER_MATCH_STATS_TABLE}
                WHERE {where_clause}
                GROUP BY role""",
            params,
        ).fetchall()
        total = sum(r["c"] for r in roles) or 1
        for r in roles:
            idx = ROLE_INDEX.get(r["role"])
            if idx is not None:
                vec[idx] = float(r["c"]) / float(total)
    finally:
        conn.close()
    return vec
