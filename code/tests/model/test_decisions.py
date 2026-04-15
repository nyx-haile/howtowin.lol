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
