#!/usr/bin/env python3
"""JSON bridge for SvelteKit. Reads JSON from stdin, writes JSON to stdout."""
import sys
import json
import os

LESSON_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "lesson_cache",
)


def get_conn(db_path=None):
    # Imported lazily so the lesson action doesn't need the legacy DB module.
    from db import get_conn as _get_conn
    return _get_conn(db_path)


def search_players(query, db_path=None):
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT puuid, riot_id, rank_tier, rank_division FROM players WHERE riot_id LIKE ? LIMIT 20",
        (f"%{query}%",)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_player_review(puuid, db_path=None):
    """Get a player's recent games with key moments — the 'lesson' data."""
    conn = get_conn(db_path)

    player = conn.execute("SELECT * FROM players WHERE puuid = ?", (puuid,)).fetchone()
    if not player:
        conn.close()
        return None

    # Recent games
    games = conn.execute("""
        SELECT DISTINCT f.match_id, g.winning_team, f.team_id, f.role,
               g.game_duration_s, g.patch
        FROM frames f
        JOIN games g ON f.match_id = g.match_id
        WHERE f.puuid = ?
        ORDER BY g.created_at DESC LIMIT 20
    """, (puuid,)).fetchall()

    game_list = []
    for g in games:
        won = g['winning_team'] == g['team_id']
        moments = conn.execute("""
            SELECT km.*, c.name as concept_name, c.explanation, c.category
            FROM key_moments km
            JOIN concepts c ON km.concept_id = c.concept_id
            WHERE km.match_id = ? AND km.team_id = ?
            ORDER BY km.impact_score DESC LIMIT 5
        """, (g['match_id'], g['team_id'])).fetchall()

        game_list.append({
            'match_id': g['match_id'],
            'role': g['role'],
            'win': won,
            'duration_s': g['game_duration_s'],
            'key_moments': [dict(m) for m in moments],
        })

    # Top concepts the player should learn (most impactful across their games)
    top_concepts = conn.execute("""
        SELECT c.concept_id, c.name, c.explanation, c.category, c.importance,
               COUNT(*) as times_relevant
        FROM key_moments km
        JOIN concepts c ON km.concept_id = c.concept_id
        JOIN frames f ON km.match_id = f.match_id AND km.team_id = f.team_id
        WHERE f.puuid = ?
        GROUP BY c.concept_id
        ORDER BY c.importance * COUNT(*) DESC
        LIMIT 5
    """, (puuid,)).fetchall()

    conn.close()

    return {
        'player': dict(player),
        'games': game_list,
        'top_concepts': [dict(c) for c in top_concepts],
    }


def get_game_review(match_id, puuid, db_path=None):
    """Get full game review data — the post-game 'lesson'."""
    conn = get_conn(db_path)

    game = conn.execute("SELECT * FROM games WHERE match_id = ?", (match_id,)).fetchone()
    if not game:
        conn.close()
        return None

    player_frame = conn.execute(
        "SELECT team_id, role FROM frames WHERE match_id = ? AND puuid = ? LIMIT 1",
        (match_id, puuid)
    ).fetchone()

    if not player_frame:
        conn.close()
        return None

    team_id = player_frame['team_id']

    # Key moments for this game, this team
    moments = conn.execute("""
        SELECT km.*, c.name as concept_name, c.explanation, c.category
        FROM key_moments km
        JOIN concepts c ON km.concept_id = c.concept_id
        WHERE km.match_id = ? AND km.team_id = ?
        ORDER BY km.timestamp_ms
    """, (match_id, team_id)).fetchall()

    # Player's frames for stat timeline
    frames = conn.execute("""
        SELECT * FROM frames
        WHERE match_id = ? AND puuid = ?
        ORDER BY timestamp_ms
    """, (match_id, puuid)).fetchall()

    # Objective events for timeline context
    objectives = conn.execute("""
        SELECT * FROM events
        WHERE match_id = ? AND event_type IN
            ('ELITE_MONSTER_KILL', 'BUILDING_KILL', 'GAME_END')
        ORDER BY timestamp_ms
    """, (match_id,)).fetchall()

    conn.close()

    won = game['winning_team'] == team_id

    return {
        'game': dict(game),
        'won': won,
        'team_id': team_id,
        'role': player_frame['role'],
        'moments': [dict(m) for m in moments],
        'frames': [dict(f) for f in frames],
        'objectives': [dict(o) for o in objectives],
    }


def _file_fingerprint(path):
    """Cheap identity for a checkpoint/index file: path + size + mtime.

    Computed without loading the file, so a cache hit never pays the
    multi-GB index load. Any retrain or index rebuild changes it.
    """
    st = os.stat(path)
    return f"{os.path.abspath(path)}:{st.st_size}:{int(st.st_mtime)}"


def get_lesson(match_id, team="blue", k=64, force=False, skill_aware=False):
    """Run model.lesson.generate_lesson, cached per (match_id, team, k).

    Cache lives at data/lesson_cache/<match_id>_<team>_k<k>.json and is
    invalidated when the checkpoint or retrieval index file changes.
    """
    if team not in ("blue", "red"):
        return {"error": f"team must be 'blue' or 'red', got {team!r}"}
    if not match_id or "/" in match_id or ".." in match_id:
        return {"error": f"invalid match_id {match_id!r}"}

    from model.lesson import DEFAULT_CHECKPOINT_PATH
    from model.retrieval import DEFAULT_INDEX_PATH

    missing = [p for p in (DEFAULT_CHECKPOINT_PATH, DEFAULT_INDEX_PATH)
               if not os.path.exists(p)]
    if missing:
        return {
            "error": "model_unavailable",
            "detail": "missing " + ", ".join(os.path.normpath(p) for p in missing),
        }

    fingerprint = {
        "checkpoint": _file_fingerprint(DEFAULT_CHECKPOINT_PATH),
        "index": _file_fingerprint(DEFAULT_INDEX_PATH),
        "skill_aware": bool(skill_aware),
    }
    os.makedirs(LESSON_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(LESSON_CACHE_DIR, f"{match_id}_{team}_k{int(k)}.json")

    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fingerprint:
                return cached["lesson"]
        except (OSError, ValueError, KeyError):
            pass  # corrupt or stale cache entry: regenerate

    from model.lesson import generate_lesson, lesson_to_dict
    lesson = lesson_to_dict(generate_lesson(
        match_id, team=team, k=int(k), skill_aware=bool(skill_aware),
    ))

    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"fingerprint": fingerprint, "lesson": lesson}, f)
    os.replace(tmp, cache_path)
    return lesson


def main():
    raw = sys.stdin.read()
    req = json.loads(raw)
    action = req['action']

    if action == 'search':
        result = search_players(req['query'])
    elif action == 'review':
        result = get_player_review(req['puuid'])
    elif action == 'game':
        result = get_game_review(req['match_id'], req['puuid'])
    elif action == 'lesson':
        # generate_lesson prints progress to stdout; keep stdout pure JSON.
        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            result = get_lesson(
                req['match_id'], team=req.get('team', 'blue'),
                k=req.get('k', 64), force=req.get('force', False),
                skill_aware=req.get('skill_aware', False),
            )
        except Exception as e:  # surface as JSON so the site can render it
            result = {'error': 'lesson_failed', 'detail': f'{type(e).__name__}: {e}'}
        finally:
            sys.stdout = real_stdout
    else:
        result = {'error': f'Unknown action: {action}'}

    json.dump(result, sys.stdout, default=str)


if __name__ == "__main__":
    main()
