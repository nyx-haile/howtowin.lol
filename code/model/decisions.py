"""Infer decision events (RECALL, ENGAGE, DISENGAGE) from frames + events.

The Riot timeline does not emit explicit "recall" events — players just
teleport home. We infer it by detecting a position jump into the team's
fountain region between consecutive frames.

ENGAGE/DISENGAGE are inferred from team-position clustering around
CHAMPION_KILL events (Task 4).
"""

BLUE_FOUNTAIN = (400, 400)
RED_FOUNTAIN = (14300, 14300)
FOUNTAIN_RADIUS = 1500  # generous; covers spawn pad + inhibitor area


def _in_fountain(pos_x, pos_y, team_id):
    fx, fy = BLUE_FOUNTAIN if team_id == 100 else RED_FOUNTAIN
    dx = pos_x - fx
    dy = pos_y - fy
    return (dx * dx + dy * dy) <= (FOUNTAIN_RADIUS * FOUNTAIN_RADIUS)


def infer_recalls(frames_by_ts):
    """Return list of {slot, timestamp_ms} for recalls.

    frames_by_ts: dict mapping timestamp_ms -> list of
                  {slot, team_id, pos_x, pos_y}.
    A recall fires when a participant moves INTO their team's fountain
    between timestamp t-1 and t.
    """
    sorted_ts = sorted(frames_by_ts.keys())
    prev_positions = {}  # slot -> (pos_x, pos_y, team_id, in_fountain)
    recalls = []

    for ts in sorted_ts:
        for f in frames_by_ts[ts]:
            slot = f["slot"]
            tid = f["team_id"]
            px, py = f["pos_x"], f["pos_y"]
            now_in = _in_fountain(px, py, tid)
            prev = prev_positions.get(slot)
            if prev is not None:
                _, _, _, prev_in = prev
                if now_in and not prev_in:
                    recalls.append({"slot": slot, "timestamp_ms": ts, "team_id": tid})
            prev_positions[slot] = (px, py, tid, now_in)

    return recalls
