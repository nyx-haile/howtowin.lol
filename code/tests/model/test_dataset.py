import sqlite3

import torch

from model.dataset import MACRO_FEAT_DIM, FRAME_FEAT_DIM, MatchDataset, collate_games, load_split
from model.static_features import STATIC_VECTOR_DIM


def _write_minimal_split_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE games (match_id TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO games (match_id) VALUES (?)",
            [("m_holdout",), ("m_cold",), ("m_train",)],
        )
        conn.commit()
    finally:
        conn.close()


def _file_snapshot(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def test_split_loads():
    holdout = load_split("holdout")
    assert isinstance(holdout, list)
    train = load_split("train")
    assert len(set(holdout) & set(train)) == 0


def test_splits_read_from_readonly_input_root(monkeypatch, tmp_path):
    fake_input = tmp_path / "input"
    split_dir = fake_input / "data" / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "plan_a_holdout.txt").write_text("m_holdout\n")
    (split_dir / "plan_b_cold_holdout.txt").write_text("m_cold\n")
    db_path = fake_input / "data" / "howtowin.db"
    _write_minimal_split_db(db_path)

    monkeypatch.setenv("HOWL_INPUT_ROOT", str(fake_input))
    monkeypatch.setenv("HOWL_DB_PATH", str(db_path))
    monkeypatch.delenv("HOWL_OUTPUT_ROOT", raising=False)

    before_files = _file_snapshot(fake_input)
    read_only_paths = [db_path, split_dir / "plan_a_holdout.txt", split_dir / "plan_b_cold_holdout.txt"]
    read_only_dirs = [split_dir, split_dir.parent, fake_input]
    try:
        for path in read_only_paths:
            path.chmod(0o444)
        for directory in read_only_dirs:
            directory.chmod(0o555)

        assert load_split("holdout") == ["m_holdout"]
        assert load_split("cold") == ["m_cold"]
        assert load_split("train") == ["m_train"]
        assert _file_snapshot(fake_input) == before_files
    finally:
        for directory in reversed(read_only_dirs):
            directory.chmod(0o755)
        for path in read_only_paths:
            path.chmod(0o644)


def test_dataset_yields_sample(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    assert sample["static"].shape == (STATIC_VECTOR_DIM,)
    assert sample["players"].shape[0] == 10
    assert sample["tokens"].dim() == 1
    assert sample["tokens"].shape == sample["token_actors"].shape == sample["token_timestamps"].shape
    assert sample["token_targets"].shape == sample["tokens"].shape
    assert sample["frame_features"].shape[-1] == FRAME_FEAT_DIM
    assert sample["anchor_macro_features"].shape[-1] == MACRO_FEAT_DIM


def test_labels_align_with_anchors(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    from model.tokens import ANCHOR_TOKEN
    is_anchor = sample["tokens"] == ANCHOR_TOKEN
    mask = sample["label_mask"].bool()
    assert mask.sum() <= is_anchor.sum()


def test_richer_anchor_features_present(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    assert sample["frame_features"].shape[-1] > 6
    assert sample["anchor_macro_features"].shape[-1] >= 6
    # current_gold / jungle_cs / kda channels should contribute non-zero signal in a real game.
    assert torch.from_numpy(sample["frame_features"]).abs().sum().item() > 0
    assert torch.from_numpy(sample["anchor_macro_features"]).abs().sum().item() > 0


def test_collate_batches_games(fixture_match_id):
    ds = MatchDataset([fixture_match_id, fixture_match_id])
    batch = collate_games([ds[0], ds[1]])
    assert batch["static"].shape == (2, STATIC_VECTOR_DIM)
    assert batch["players"].shape[0] == 2
    assert batch["tokens"].shape[0] == 2
    assert batch["anchor_mask"].shape[:1] == (2,)
    assert batch["anchor_macro_features"].shape[0] == 2
    assert batch["event_window_positions"].dtype == torch.long
    assert batch["tokens"].shape[1] == batch["labels"].shape[1]
