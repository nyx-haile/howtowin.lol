import hashlib
import sqlite3
from pathlib import Path

from model.cold_holdout import (
    build_player_cold_holdout, COLD_HOLDOUT_PATH,
    PLAYER_COLD_GAMES_FRACTION, PLAYER_COLD_TOLERANCE,
)
from db import get_conn


def test_build_is_deterministic():
    a = build_player_cold_holdout()
    b = build_player_cold_holdout()
    assert a == b


def test_cold_games_fraction_within_tolerance():
    puuids, match_ids = build_player_cold_holdout()
    conn = get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    finally:
        conn.close()
    if total == 0:
        return
    frac = len(match_ids) / total
    # Hit at or above target, within upper tolerance.
    assert frac >= PLAYER_COLD_GAMES_FRACTION * 0.9, (
        f"cold fraction {frac:.3f} below target {PLAYER_COLD_GAMES_FRACTION}"
    )
    assert frac <= PLAYER_COLD_GAMES_FRACTION * (1 + PLAYER_COLD_TOLERANCE) * 1.5, (
        f"cold fraction {frac:.3f} exceeds tolerance"
    )


def test_match_ids_actually_contain_those_puuids():
    from db import get_conn
    puuids, match_ids = build_player_cold_holdout()
    conn = get_conn()
    for mid in list(match_ids)[:5]:
        rows = conn.execute(
            "SELECT DISTINCT puuid FROM frames WHERE match_id = ?", (mid,)
        ).fetchall()
        participants = {r["puuid"] for r in rows}
        assert participants & puuids, f"cold match {mid} has no cold puuid"
    conn.close()


def test_no_overlap_with_game_cold_holdout():
    from model.dataset import load_split
    game_cold = set(load_split("holdout"))
    _puuids, cold_match_ids = build_player_cold_holdout()
    overlap = game_cold & cold_match_ids
    assert len(cold_match_ids) > len(overlap), (
        "player-cold holdout adds no new held-out games beyond game-cold"
    )


def test_constant_path_points_into_splits_dir():
    assert "data/splits/plan_b_cold_holdout.txt" in str(COLD_HOLDOUT_PATH)


# ---------------------------------------------------------------------------
# Progress logging (bead 1mz) — the k-expansion / trim search issues one DB
# query per step and used to run entirely silent.
# ---------------------------------------------------------------------------


def test_build_player_cold_holdout_emits_progress(capsys):
    puuids, matches = build_player_cold_holdout()
    out = capsys.readouterr().out
    if not puuids and not matches:
        # Empty corpus — nothing to report on.
        return
    assert "[cold-holdout] player-cold: puuids=" in out, out
    assert "[cold-holdout] expand step 1: k=" in out, out
    assert "[cold-holdout] player-cold done: puuids=" in out, out


def _write_minimal_holdout_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE games (match_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE frames (match_id TEXT, puuid TEXT)")
        for index in range(4):
            match_id = f"m{index}"
            conn.execute("INSERT INTO games (match_id) VALUES (?)", (match_id,))
            conn.execute(
                "INSERT INTO frames (match_id, puuid) VALUES (?, ?)",
                (match_id, f"puuid-{index}"),
            )
        conn.commit()
    finally:
        conn.close()


def _file_snapshot(root: Path):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def test_generation_writes_splits_to_output_root_only(monkeypatch, tmp_path):
    from model import cold_holdout as ch

    fake_input = tmp_path / "input"
    fake_output = tmp_path / "working" / "howtowin"
    input_data = fake_input / "data"
    input_data.mkdir(parents=True)
    db_path = input_data / "howtowin.db"
    _write_minimal_holdout_db(db_path)

    monkeypatch.setenv("HOWL_INPUT_ROOT", str(fake_input))
    monkeypatch.setenv("HOWL_OUTPUT_ROOT", str(fake_output))
    monkeypatch.setenv("HOWL_DB_PATH", str(db_path))

    before_files = _file_snapshot(fake_input)
    read_only_paths = [db_path]
    read_only_dirs = [input_data, fake_input]
    try:
        for path in read_only_paths:
            path.chmod(0o444)
        for directory in read_only_dirs:
            directory.chmod(0o555)

        player_cold_n, game_cold_n, train_n = ch.rebuild_all_splits(
            player_fraction=0.25,
            game_fraction=1.0,
        )

        output_split_dir = fake_output / "data" / "splits"
        assert (output_split_dir / "plan_b_cold_holdout.txt").exists()
        assert (output_split_dir / "plan_a_holdout.txt").exists()
        assert Path(ch.generated_cold_holdout_path()).is_relative_to(fake_output)
        assert Path(ch.generated_game_holdout_path()).is_relative_to(fake_output)
        assert player_cold_n >= 1
        assert game_cold_n + player_cold_n + train_n == 4
        assert _file_snapshot(fake_input) == before_files
    finally:
        for directory in reversed(read_only_dirs):
            directory.chmod(0o755)
        for path in read_only_paths:
            path.chmod(0o644)
