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


FIGHT_RADIUS = 2500
PRE_FIGHT_WINDOW_MS = 15000
MIN_ATTACKERS = 3
MIN_DEFENDERS = 2


def _nearest_frame_ts(frames_by_ts, target_ts):
    """Return the frame timestamp closest to target_ts but not after it."""
    candidates = [ts for ts in frames_by_ts.keys() if ts <= target_ts]
    if not candidates:
        return None
    return max(candidates)


def _count_nearby(frames, cx, cy, team_id):
    n = 0
    for f in frames:
        if f["team_id"] != team_id:
            continue
        dx = f["pos_x"] - cx
        dy = f["pos_y"] - cy
        if dx * dx + dy * dy <= FIGHT_RADIUS * FIGHT_RADIUS:
            n += 1
    return n


def infer_engages(kill_events, frames_by_ts):
    """Return list of {event_type: ENGAGE|DISENGAGE, team_id, timestamp_ms}.

    kill_events: list of {timestamp_ms, killer_id, victim_id, position_x, position_y}.
    frames_by_ts: same shape as infer_recalls.

    Heuristic: for each kill, look at frames ~PRE_FIGHT_WINDOW_MS before the kill.
    Count attackers and defenders within FIGHT_RADIUS of the kill position.
    If the attacker count >= MIN_ATTACKERS and defender count >= MIN_DEFENDERS,
    emit an ENGAGE tagged to the attacker team. If defender count exceeds
    attacker count at that moment, also emit a DISENGAGE on the defender team
    (the defending team tried to contest and got killed).
    """
    seen_team_ts = set()  # dedupe: one engage per team per kill-timestamp
    tags = []

    for ev in kill_events:
        kts = ev["timestamp_ms"]
        pre_ts = _nearest_frame_ts(frames_by_ts, kts - PRE_FIGHT_WINDOW_MS)
        if pre_ts is None:
            continue
        frames = frames_by_ts[pre_ts]

        killer_slot = ev.get("killer_id", 0)
        victim_slot = ev.get("victim_id", 0)
        if not (1 <= killer_slot <= 10) or not (1 <= victim_slot <= 10):
            continue

        killer_team = 100 if killer_slot <= 5 else 200
        victim_team = 100 if victim_slot <= 5 else 200
        if killer_team == victim_team:
            continue  # jungle monster kills etc.

        cx, cy = ev["position_x"], ev["position_y"]
        attackers = _count_nearby(frames, cx, cy, killer_team)
        defenders = _count_nearby(frames, cx, cy, victim_team)

        if attackers >= MIN_ATTACKERS and defenders >= MIN_DEFENDERS:
            key = ("ENGAGE", killer_team, kts)
            if key not in seen_team_ts:
                tags.append({"event_type": "ENGAGE", "team_id": killer_team, "timestamp_ms": kts})
                seen_team_ts.add(key)
            if defenders > attackers:
                key2 = ("DISENGAGE", victim_team, kts)
                if key2 not in seen_team_ts:
                    tags.append({"event_type": "DISENGAGE", "team_id": victim_team, "timestamp_ms": kts})
                    seen_team_ts.add(key2)

    return tags
