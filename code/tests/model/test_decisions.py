from model.decisions import infer_recalls, FOUNTAIN_RADIUS, BLUE_FOUNTAIN, RED_FOUNTAIN


def test_recall_detected_on_fountain_jump():
    frames_by_ts = {
        60000: [
            {"slot": 1, "team_id": 100, "pos_x": 7000, "pos_y": 7000},
        ],
        120000: [
            {"slot": 1, "team_id": 100, "pos_x": BLUE_FOUNTAIN[0] + 100, "pos_y": BLUE_FOUNTAIN[1] + 100},
        ],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 1
    assert recalls[0]["slot"] == 1
    assert recalls[0]["timestamp_ms"] == 120000


def test_no_recall_when_already_at_fountain():
    frames_by_ts = {
        60000: [
            {"slot": 1, "team_id": 100, "pos_x": BLUE_FOUNTAIN[0], "pos_y": BLUE_FOUNTAIN[1]},
        ],
        120000: [
            {"slot": 1, "team_id": 100, "pos_x": BLUE_FOUNTAIN[0] + 50, "pos_y": BLUE_FOUNTAIN[1] + 50},
        ],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 0


def test_red_team_uses_red_fountain():
    frames_by_ts = {
        60000: [{"slot": 6, "team_id": 200, "pos_x": 7000, "pos_y": 7000}],
        120000: [{"slot": 6, "team_id": 200, "pos_x": RED_FOUNTAIN[0] - 100, "pos_y": RED_FOUNTAIN[1] - 100}],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 1
    assert recalls[0]["slot"] == 6


def test_red_at_blue_fountain_is_not_recall():
    frames_by_ts = {
        60000: [{"slot": 6, "team_id": 200, "pos_x": 7000, "pos_y": 7000}],
        120000: [{"slot": 6, "team_id": 200, "pos_x": BLUE_FOUNTAIN[0] + 50, "pos_y": BLUE_FOUNTAIN[1] + 50}],
    }
    recalls = infer_recalls(frames_by_ts)
    assert len(recalls) == 0


def test_fountain_radius_positive():
    assert FOUNTAIN_RADIUS > 0


from model.decisions import infer_engages, FIGHT_RADIUS, PRE_FIGHT_WINDOW_MS


def test_engage_detected_on_clustered_kill():
    # CHAMPION_KILL at t=300000 with killer slot 1, victim slot 6.
    # At t=285000 (15s pre-fight), 3 blue members and 2 red members clustered.
    frames_by_ts = {
        285000: [
            {"slot": 1, "team_id": 100, "pos_x": 7500, "pos_y": 7500},
            {"slot": 2, "team_id": 100, "pos_x": 7600, "pos_y": 7500},
            {"slot": 3, "team_id": 100, "pos_x": 7500, "pos_y": 7600},
            {"slot": 6, "team_id": 200, "pos_x": 7700, "pos_y": 7700},
            {"slot": 7, "team_id": 200, "pos_x": 7800, "pos_y": 7700},
        ],
    }
    kill_events = [
        {"timestamp_ms": 300000, "killer_id": 1, "victim_id": 6,
         "position_x": 7700, "position_y": 7700},
    ]
    tags = infer_engages(kill_events, frames_by_ts)
    engages = [t for t in tags if t["event_type"] == "ENGAGE"]
    assert len(engages) == 1
    assert engages[0]["team_id"] == 100


def test_no_engage_when_positions_far():
    # Only one blue nearby — not an engage, just a pick.
    frames_by_ts = {
        285000: [
            {"slot": 1, "team_id": 100, "pos_x": 7500, "pos_y": 7500},
            {"slot": 6, "team_id": 200, "pos_x": 7700, "pos_y": 7700},
        ],
    }
    kill_events = [
        {"timestamp_ms": 300000, "killer_id": 1, "victim_id": 6,
         "position_x": 7700, "position_y": 7700},
    ]
    tags = infer_engages(kill_events, frames_by_ts)
    assert len(tags) == 0


def test_fight_radius_and_window_positive():
    assert FIGHT_RADIUS > 0
    assert PRE_FIGHT_WINDOW_MS > 0
