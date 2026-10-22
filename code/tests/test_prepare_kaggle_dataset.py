from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "prepare-kaggle-dataset.py"


def load_packager_module():
    spec = importlib.util.spec_from_file_location("prepare_kaggle_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_sqlite_db(path: Path, table_name: str = "items") -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute(f"INSERT INTO {table_name} (value) VALUES (?)", ("fixture",))
        conn.commit()
    finally:
        conn.close()


def build_fixture_tree(root: Path) -> Path:
    data_root = root / "data"
    data_root.mkdir(parents=True)
    write_sqlite_db(data_root / "howtowin.db")
    write_sqlite_db(data_root / "raw_matches.db", table_name="raw_items")
    splits = data_root / "splits"
    splits.mkdir()
    (splits / "plan_a_holdout.txt").write_text("m1\n", encoding="utf-8")
    (splits / "plan_b_cold_holdout.txt").write_text("m2\n", encoding="utf-8")
    (data_root / "materialized_samples").mkdir()
    (data_root / "materialized_samples" / "sample.pt").write_bytes(b"cache")
    (data_root / "retrieval").mkdir()
    (data_root / "retrieval" / "cache").mkdir(parents=True, exist_ok=True)
    (data_root / "retrieval" / "cache" / "index.pt").write_bytes(b"index")
    (data_root / "model_checkpoints").mkdir()
    (data_root / "model_checkpoints" / "checkpoint.pt").write_bytes(b"checkpoint")
    (data_root / "logs").mkdir()
    (data_root / "logs" / "run.log").write_text("log\n", encoding="utf-8")
    (data_root / "reports").mkdir()
    (data_root / "reports" / "report.md").write_text("report\n", encoding="utf-8")
    (data_root / "artifact.bin").write_bytes(b"nope")
    return data_root


def test_manifest_includes_expected_files_and_excludes_generated_artifacts(tmp_path):
    module = load_packager_module()
    data_root = build_fixture_tree(tmp_path)
    staging_dir = tmp_path / "staging"

    manifest = module.build_manifest(data_root, staging_dir)
    module.write_manifest_bundle(manifest, staging_dir)

    manifest_json_path = staging_dir / "manifest.json"
    manifest_md_path = staging_dir / "manifest.md"
    readme_path = staging_dir / "README.md"
    checksums_path = staging_dir / "checksums.txt"

    assert manifest_json_path.exists()
    assert manifest_md_path.exists()
    assert readme_path.exists()
    assert checksums_path.exists()
    assert (staging_dir / "data" / "howtowin.db").exists()
    assert (staging_dir / "data" / "raw_matches.db").exists()
    assert (staging_dir / "data" / "splits" / "plan_a_holdout.txt").exists()
    assert (staging_dir / "data" / "splits" / "plan_b_cold_holdout.txt").exists()

    loaded = json.loads(manifest_json_path.read_text(encoding="utf-8"))
    included_paths = {entry["path"] for entry in loaded["included_files"]}
    excluded_paths = {entry["path"] for entry in loaded["excluded_files"]}

    assert loaded["status"] == "ok"
    assert loaded["payload_dir"] == str(staging_dir.resolve() / "data")
    assert loaded["payload_strategy"] == "auto"
    assert included_paths == {
        "howtowin.db",
        "raw_matches.db",
        "splits/plan_a_holdout.txt",
        "splits/plan_b_cold_holdout.txt",
    }
    assert "materialized_samples/sample.pt" in excluded_paths
    assert "retrieval/cache/index.pt" in excluded_paths
    assert "model_checkpoints/checkpoint.pt" in excluded_paths
    assert "logs/run.log" in excluded_paths
    assert "reports/report.md" in excluded_paths
    assert "artifact.bin" in excluded_paths

    expected_total = sum(entry["size_bytes"] for entry in loaded["included_files"])
    assert loaded["total_bytes"] == expected_total
    assert loaded["sqlite_quick_check"] == [
        {"path": "howtowin.db", "quick_check": "ok"},
        {"path": "raw_matches.db", "quick_check": "ok"},
    ]
    assert "howtowin.db" in manifest_md_path.read_text(encoding="utf-8")
    assert "excluded generated directory" in manifest_md_path.read_text(encoding="utf-8")
    assert "data/howtowin.db" in readme_path.read_text(encoding="utf-8")
    assert checksums_path.read_text(encoding="utf-8").count("\n") == len(included_paths)


@pytest.mark.parametrize("sidecar_suffix", ["-wal", "-shm"])
def test_hot_wal_and_shm_sidecars_are_rejected(tmp_path, sidecar_suffix):
    module = load_packager_module()
    data_root = tmp_path / "data"
    data_root.mkdir()
    write_sqlite_db(data_root / "howtowin.db")
    write_sqlite_db(data_root / "raw_matches.db", table_name="raw_items")
    (data_root / "splits").mkdir()
    (data_root / "splits" / "plan_a_holdout.txt").write_text("m1\n", encoding="utf-8")

    sidecar = data_root / f"howtowin.db{sidecar_suffix}"
    sidecar.write_bytes(b"hot")

    with pytest.raises(SystemExit, match="hot SQLite sidecar detected"):
        module.build_manifest(data_root, tmp_path / "staging")


def test_total_size_limit_is_enforced(monkeypatch, tmp_path):
    module = load_packager_module()
    data_root = build_fixture_tree(tmp_path)

    def fake_size(path: Path) -> int:
        if path.name == "howtowin.db":
            return module.TOTAL_SIZE_LIMIT_BYTES + 1
        return path.stat().st_size

    monkeypatch.setattr(module, "file_size_bytes", fake_size)

    with pytest.raises(SystemExit, match="exceeds Kaggle limit"):
        module.build_manifest(data_root, tmp_path / "staging")
