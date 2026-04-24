"""Holdout builders for game-cold and player-cold evaluation.

Two disjoint holdouts held out from training:

1. player-cold (plan_b_cold_holdout.txt) — every match in which any of the
   top-K puuids (by SHA1 hash rank) participated. Tests generalisation to
   unseen players.
2. game-cold  (plan_a_holdout.txt) — a deterministic uniform-hash sample
   of the remaining matches. Tests generalisation to unseen matches by
   already-seen players.

Both sizes are governed by fractions of the total corpus. Defaults aim for
~10% each (so ~80% train).

Plan B spec section B2.
"""
import hashlib
import os
from db import get_conn

SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits")
COLD_HOLDOUT_PATH = os.path.join(SPLIT_DIR, "plan_b_cold_holdout.txt")
GAME_HOLDOUT_PATH = os.path.join(SPLIT_DIR, "plan_a_holdout.txt")

# Target fraction of total matches in each holdout. Train = 1 - 2*fraction.
PLAYER_COLD_GAMES_FRACTION = 0.10
GAME_COLD_FRACTION = 0.10

# Overshoot tolerance when growing the puuid set. Stop once cold fraction
# falls within [target, target * (1 + PLAYER_COLD_TOLERANCE)].
PLAYER_COLD_TOLERANCE = 0.10


def _sha1_rank(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _sha1_int(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def _fetch_all_puuids(conn):
    rows = conn.execute("SELECT DISTINCT puuid FROM frames").fetchall()
    return [r["puuid"] for r in rows]


def _fetch_all_match_ids(conn):
    rows = conn.execute("SELECT match_id FROM games").fetchall()
    return [r["match_id"] for r in rows]


def _matches_for_puuids(conn, puuids):
    if not puuids:
        return set()
    placeholders = ",".join("?" * len(puuids))
    rows = conn.execute(
        f"SELECT DISTINCT match_id FROM frames WHERE puuid IN ({placeholders})",
        tuple(puuids),
    ).fetchall()
    return {r["match_id"] for r in rows}


def build_player_cold_holdout(target_fraction: float = PLAYER_COLD_GAMES_FRACTION) -> tuple[set[str], set[str]]:
    """Pick top-K puuids by SHA1 hash rank such that the resulting cold-game
    set is approximately `target_fraction` of all matches. K is scaled
    iteratively to reach the target without massive overshoot.
    """
    conn = get_conn()
    try:
        all_puuids = _fetch_all_puuids(conn)
        all_puuids.sort(key=_sha1_rank)
        total_matches = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        if total_matches == 0 or not all_puuids:
            return set(), set()

        target_games = int(total_matches * target_fraction)
        upper_games = int(total_matches * target_fraction * (1 + PLAYER_COLD_TOLERANCE))

        # Binary-ish expansion — exponential growth then linear trim.
        lo, hi = 1, len(all_puuids)
        chosen_k = hi
        matches: set[str] = set()

        k = max(1, target_games // 20)  # seed: assume ~20 games/puuid average
        while k <= hi:
            candidate = set(all_puuids[:k])
            m = _matches_for_puuids(conn, list(candidate))
            if len(m) >= target_games:
                chosen_k = k
                matches = m
                break
            k = max(k + 1, int(k * 1.5))
        else:
            # Did not reach target — use all puuids (should not happen)
            chosen_k = hi
            matches = _matches_for_puuids(conn, list(all_puuids))

        # Trim back if over upper bound.
        if len(matches) > upper_games and chosen_k > 1:
            while chosen_k > 1:
                chosen_k -= 1
                candidate = set(all_puuids[:chosen_k])
                m = _matches_for_puuids(conn, list(candidate))
                if len(m) < target_games:
                    # Went too far; step back up.
                    chosen_k += 1
                    candidate = set(all_puuids[:chosen_k])
                    matches = _matches_for_puuids(conn, list(candidate))
                    break
                matches = m

        chosen_puuids = set(all_puuids[:chosen_k])
    finally:
        conn.close()
    return chosen_puuids, matches


def build_game_cold_holdout(
    exclude: set[str],
    fraction: float = GAME_COLD_FRACTION,
) -> set[str]:
    """Deterministic uniform sample via SHA1(match_id) mod N. `exclude`
    typically contains the player-cold match ids; we sample from the rest.
    """
    conn = get_conn()
    try:
        all_ids = _fetch_all_match_ids(conn)
    finally:
        conn.close()
    candidates = [m for m in all_ids if m not in exclude]
    if not candidates or fraction <= 0:
        return set()
    threshold = int(fraction * (1 << 32))
    chosen = {m for m in candidates if (_sha1_int(m) & 0xFFFFFFFF) < threshold}
    return chosen


def save_player_cold_holdout(
    path: str = COLD_HOLDOUT_PATH,
    target_fraction: float = PLAYER_COLD_GAMES_FRACTION,
) -> tuple[set[str], set[str]]:
    puuids, match_ids = build_player_cold_holdout(target_fraction=target_fraction)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for mid in sorted(match_ids):
            f.write(mid + "\n")
    return puuids, match_ids


def save_game_cold_holdout(
    path: str = GAME_HOLDOUT_PATH,
    exclude: set[str] | None = None,
    fraction: float = GAME_COLD_FRACTION,
) -> set[str]:
    exclude = exclude or set()
    match_ids = build_game_cold_holdout(exclude=exclude, fraction=fraction)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for mid in sorted(match_ids):
            f.write(mid + "\n")
    return match_ids


def load_player_cold_holdout(path: str | None = None) -> set[str]:
    path = path or COLD_HOLDOUT_PATH
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_game_cold_holdout(path: str | None = None) -> set[str]:
    path = path or GAME_HOLDOUT_PATH
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def rebuild_all_splits(
    player_fraction: float = PLAYER_COLD_GAMES_FRACTION,
    game_fraction: float = GAME_COLD_FRACTION,
) -> tuple[int, int, int]:
    """Regenerate both split files. Returns (player_cold_n, game_cold_n, train_n)."""
    puuids, player_cold = save_player_cold_holdout(target_fraction=player_fraction)
    game_cold = save_game_cold_holdout(exclude=player_cold, fraction=game_fraction)
    conn = get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    finally:
        conn.close()
    train_n = total - len(player_cold) - len(game_cold)
    print(
        f"splits rebuilt: player_cold={len(player_cold)} "
        f"({len(puuids)} puuids), game_cold={len(game_cold)}, "
        f"train={train_n}, total={total}"
    )
    return len(player_cold), len(game_cold), train_n


if __name__ == "__main__":
    rebuild_all_splits()
