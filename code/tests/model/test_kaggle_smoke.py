import json
import os
import subprocess
import sys
from pathlib import Path

import db
import raw_db


def _make_main_db(path: Path) -> None:
    db.init_db(path)
    conn = db.get_conn(path)
    try:
        for match_id, winning_team in (("NA1_smoke_train", 100), ("NA1_smoke_holdout", 200)):
            db.insert_game(conn, match_id, "14.15", 420, 1800, winning_team, 1700000000)
            for ts_ms in (0, 60000, 120000):
                for slot in range(1, 11):
                    team_id = 100 if slot <= 5 else 200
                    db.insert_frame(
                        conn,
                        match_id,
                        slot,
                        f"{match_id}_puuid_{slot}",
                        team_id,
                        "MID",
                        ts_ms,
                        500 + slot,
                        1000 + ts_ms // 1000 + slot,
                        800 + ts_ms // 1000 + slot,
                        1 + ts_ms // 60000,
                        10 + slot,
                        0,
                        1000 + slot,
                        2000 + slot,
                        1 if slot == 1 and ts_ms >= 60000 else 0,
                        0,
                        0,
                        1,
                        [],
                    )
            db.insert_event(
                conn,
                match_id,
                30000,
                "ITEM_PURCHASED",
                participant_id=1,
                details='{"itemId": 1055}',
            )
            db.insert_event(
                conn,
                match_id,
                90000,
                "SKILL_LEVEL_UP",
                participant_id=2,
                details='{"skillSlot": 1}',
            )
            for slot in range(1, 11):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO player_match_stats (
                        puuid, match_id, team_id, role, created_at, winning_team,
                        window_10_rows, kda_10_sum, cs_10_sum, total_gold_10_sum, ward_count_10_sum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{match_id}_puuid_{slot}",
                        match_id,
                        100 if slot <= 5 else 200,
                        "MID",
                        1700000000,
                        winning_team,
                        1,
                        1.0,
                        10.0,
                        1000.0,
                        1.0,
                    ),
                )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


def _make_raw_db(path: Path) -> None:
    raw_db.init_raw_db(path)
    conn = raw_db.get_raw_conn(path)
    try:
        for match_id in ("NA1_smoke_train", "NA1_smoke_holdout"):
            participants = [
                {"puuid": f"{match_id}_puuid_{slot}", "championId": 100 + slot, "teamId": 100 if slot <= 5 else 200}
                for slot in range(1, 11)
            ]
            match = {
                "metadata": {"matchId": match_id, "platformId": "NA1"},
                "info": {"queueId": 420, "platformId": "NA1", "gameCreation": 1700000000000, "participants": participants},
            }
            timeline = {"info": {"frames": []}}
            raw_db.insert_raw_match(conn, match_id, match, timeline)
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


def test_kaggle_smoke_uses_readonly_fixture_dbs_and_writes_only_to_working(tmp_path):
    slug = "owner/howtowin-db-v001"
    fake_input_root = tmp_path / "kaggle" / "input" / slug
    data_dir = fake_input_root / "data"
    splits_dir = data_dir / "splits"
    working_root = tmp_path / "kaggle" / "working" / "howtowin"
    splits_dir.mkdir(parents=True)

    main_db = data_dir / "howtowin.db"
    raw_matches_db = data_dir / "raw_matches.db"
    _make_main_db(main_db)
    _make_raw_db(raw_matches_db)
    (splits_dir / "plan_a_holdout.txt").write_text("NA1_smoke_holdout\n", encoding="utf-8")
    (splits_dir / "plan_b_cold_holdout.txt").write_text("", encoding="utf-8")

    before_input = sorted(p.relative_to(fake_input_root) for p in fake_input_root.rglob("*"))

    env = os.environ.copy()
    env.update(
        {
            "HOWL_DB_PATH": f"{main_db.as_uri()}?mode=ro&immutable=1",
            "HOWL_RAW_DB_PATH": f"{raw_matches_db.as_uri()}?mode=ro&immutable=1",
            "HOWL_INPUT_ROOT": str(fake_input_root),
            "HOWL_OUTPUT_ROOT": str(working_root),
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        }
    )

    script = Path(__file__).resolve().parents[3] / "scripts" / "kaggle_smoke.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--steps",
            "2",
            "--allow-local-paths",
            "--input-root",
            str(fake_input_root),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["status"] == "ok"
    assert [step["step"] for step in summary["steps"]] == ["imports", "training-smoke"]
    assert summary["steps"][1]["requested_optimizer_steps"] == 2
    assert summary["steps"][1]["train_loss_count"] == 1
    assert summary["steps"][1]["db_connectivity"]["game_count"] == 2
    assert summary["steps"][1]["db_connectivity"]["raw_match_count"] == 2

    artifact = working_root / "kaggle_smoke_result.json"
    assert artifact.exists()
    assert artifact.resolve().is_relative_to(working_root.resolve())
    assert sorted(p.relative_to(fake_input_root) for p in fake_input_root.rglob("*")) == before_input
