"""Player-cold-start holdout builder.

Deterministically selects COLD_HOLDOUT_PUUID_COUNT puuids by SHA1 hash rank,
then reserves every match containing any of those puuids.  This prevents the
model from memorizing player identities during training.

Plan B spec section B2.
"""
import hashlib
import os
from db import get_conn

COLD_HOLDOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "splits", "plan_b_cold_holdout.txt"
)
COLD_HOLDOUT_PUUID_COUNT = 5


def _sha1_rank(puuid: str) -> str:
    return hashlib.sha1(puuid.encode("utf-8")).hexdigest()


def build_player_cold_holdout() -> tuple[set[str], set[str]]:
    """Deterministically pick COLD_HOLDOUT_PUUID_COUNT puuids by SHA1 hash
    order, then collect every match those puuids appeared in.  Return
    (chosen_puuids, match_ids)."""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT DISTINCT puuid FROM frames").fetchall()
        all_puuids = [r["puuid"] for r in rows]
        all_puuids.sort(key=_sha1_rank)
        chosen_puuids = set(all_puuids[:COLD_HOLDOUT_PUUID_COUNT])

        if not chosen_puuids:
            return set(), set()

        placeholders = ",".join("?" * len(chosen_puuids))
        rows = conn.execute(
            f"SELECT DISTINCT match_id FROM frames WHERE puuid IN ({placeholders})",
            tuple(chosen_puuids),
        ).fetchall()
        match_ids = {r["match_id"] for r in rows}
    finally:
        conn.close()
    return chosen_puuids, match_ids


def save_player_cold_holdout(path: str = COLD_HOLDOUT_PATH) -> tuple[set[str], set[str]]:
    """Build the holdout and persist the match IDs to disk."""
    puuids, match_ids = build_player_cold_holdout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for mid in sorted(match_ids):
            f.write(mid + "\n")
    return puuids, match_ids


def load_player_cold_holdout(path: str = COLD_HOLDOUT_PATH) -> set[str]:
    """Load previously saved player-cold match IDs from disk."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}
