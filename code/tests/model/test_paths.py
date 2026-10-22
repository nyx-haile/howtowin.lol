from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from runtime_paths import artifact_path, resolve_runtime_paths


def test_local_defaults_preserved(monkeypatch):
    monkeypatch.delenv("HOWL_INPUT_ROOT", raising=False)
    monkeypatch.delenv("HOWL_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("HOWL_DB_PATH", raising=False)
    monkeypatch.delenv("HOWL_RAW_DB_PATH", raising=False)

    paths = resolve_runtime_paths()
    repo_root = Path(__file__).resolve().parents[3]

    assert paths.input_root == repo_root
    assert paths.output_root == repo_root
    assert paths.data_root == repo_root / "data"
    assert paths.db_path == repo_root / "data" / "howtowin.db"
    assert paths.raw_db_path == repo_root / "data" / "raw_matches.db"
    assert paths.split_dir == repo_root / "data" / "splits"
    assert paths.checkpoint_dir == repo_root / "data" / "model_checkpoints"
    assert paths.retrieval_dir == repo_root / "data" / "retrieval"
    assert paths.cache_root == repo_root / "data" / "retrieval" / "cache"
    assert paths.report_artifact_root == repo_root / "artifacts"
    assert paths.materialized_sample_root == repo_root / "data" / "materialized_samples" / "plan_b"


def test_howl_output_root_redirects_writable_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("HOWL_INPUT_ROOT", raising=False)
    monkeypatch.delenv("HOWL_DB_PATH", raising=False)
    monkeypatch.delenv("HOWL_RAW_DB_PATH", raising=False)
    monkeypatch.setenv("HOWL_OUTPUT_ROOT", str(tmp_path))

    paths = resolve_runtime_paths()
    repo_root = Path(__file__).resolve().parents[3]

    assert paths.data_root == repo_root / "data"
    assert paths.db_path == repo_root / "data" / "howtowin.db"
    assert paths.raw_db_path == repo_root / "data" / "raw_matches.db"
    assert paths.split_dir == repo_root / "data" / "splits"
    assert paths.checkpoint_dir == tmp_path / "data" / "model_checkpoints"
    assert paths.retrieval_dir == tmp_path / "data" / "retrieval"
    assert paths.cache_root == tmp_path / "data" / "retrieval" / "cache"
    assert paths.report_artifact_root == tmp_path / "artifacts"
    assert paths.materialized_sample_root == tmp_path / "data" / "materialized_samples" / "plan_b"
    assert artifact_path("probe.json") == tmp_path / "artifacts" / "probe.json"


def test_output_root_routes_model_writable_defaults_before_import(tmp_path):
    fake_input = tmp_path / "input"
    fake_output = tmp_path / "working" / "howtowin"
    split_dir = fake_input / "data" / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "plan_a_holdout.txt").write_text("fixture-match\n")
    before_input_files = sorted(
        str(path.relative_to(fake_input))
        for path in fake_input.rglob("*")
        if path.is_file()
    )

    for directory in (split_dir, split_dir.parent, fake_input):
        directory.chmod(0o555)

    script = r'''
import json
from pathlib import Path

import torch

from runtime_paths import artifact_path, resolve_runtime_paths
from model import cli, eval_cache, lesson, plan_b_train, retrieval, train
from model.sample_materialization import PlanBSampleCache

paths = resolve_runtime_paths()
preflight_path = artifact_path("runtime_preflight.json")
plan_b_train.write_runtime_preflight_artifact(str(preflight_path), {
    "projected_earlystop_hours": 0.0,
    "cache_hit_rate": 0.0,
    "gate_a_pass": True,
})

checkpoint_path = Path(plan_b_train.CHECKPOINT_DIR) / "path_probe_best.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({"ok": True}, checkpoint_path)

eval_cache.save_cached_keys(
    "model", "game_cold", checkpoint_sha="sha", holdout_match_ids=["m1"],
    keys=torch.zeros(1, retrieval.KEY_DIM), minutes=torch.tensor([10]),
)
eval_cache.save_cached_cohort_idx(
    "model", "game_cold", checkpoint_sha="sha", holdout_match_ids=["m1"],
    cohort_idx=torch.zeros(1, 1, dtype=torch.long),
)

empty_index = retrieval.IndexBundle(
    corpus_white=torch.zeros(0, retrieval.KEY_DIM),
    whitener=retrieval.Whitener(torch.zeros(retrieval.KEY_DIM), torch.ones(retrieval.KEY_DIM)),
    row_match_id=[],
    row_anchor_minute=torch.zeros(0, dtype=torch.int64),
    row_blue_win=torch.zeros(0, dtype=torch.int8),
    checkpoint_sha="sha",
    code_sha="test",
    built_at=0,
)
retrieval.save_index(empty_index, retrieval.DEFAULT_INDEX_PATH)

sample_cache = PlanBSampleCache(puuid_index={}, exclude_match_ids=set(), mode="readwrite")
sample_cache.store("m1", {"x": torch.tensor([1])})

report_path = artifact_path("m4_retrieval_eval_report_probe.md")
report_result = {
    "n_queries": 1,
    "model": {"k_sweep": {64: 0.8}, "per_minute_at_headline_k": {10: 0.8}},
}
cli._write_report(str(report_path), {"game_cold": report_result, "player_cold": report_result}, True)

print(json.dumps({
    "db_path": str(paths.db_path),
    "split_dir": str(paths.split_dir),
    "checkpoint_dir": plan_b_train.CHECKPOINT_DIR,
    "baseline_checkpoint_dir": train.CHECKPOINT_DIR,
    "lesson_checkpoint": lesson.DEFAULT_CHECKPOINT_PATH,
    "retrieval_dir": retrieval.INDEX_DIR,
    "retrieval_index": retrieval.DEFAULT_INDEX_PATH,
    "eval_cache_dir": str(eval_cache._CACHE_DIR),
    "sample_cache_dir": sample_cache.config.cache_dir,
    "preflight_path": str(preflight_path),
    "report_path": str(report_path),
}))
'''
    env = os.environ.copy()
    env.update({
        "HOWL_INPUT_ROOT": str(fake_input),
        "HOWL_OUTPUT_ROOT": str(fake_output),
    })
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    finally:
        for directory in (fake_input, split_dir.parent, split_dir):
            directory.chmod(0o755)

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    output_paths = [
        "checkpoint_dir",
        "baseline_checkpoint_dir",
        "lesson_checkpoint",
        "retrieval_dir",
        "retrieval_index",
        "eval_cache_dir",
        "sample_cache_dir",
        "preflight_path",
        "report_path",
    ]
    for key in output_paths:
        assert Path(payload[key]).is_relative_to(fake_output), (key, payload[key])
    assert Path(payload["db_path"]).is_relative_to(fake_input)
    assert Path(payload["split_dir"]).is_relative_to(fake_input)

    after_input_files = sorted(
        str(path.relative_to(fake_input))
        for path in fake_input.rglob("*")
        if path.is_file()
    )
    assert after_input_files == before_input_files
    assert any(fake_output.rglob("*.pt"))
    assert (fake_output / "artifacts" / "runtime_preflight.json").exists()


def test_explicit_and_existing_db_env_overrides_win(monkeypatch, tmp_path):
    explicit_db = tmp_path / "explicit.db"
    explicit_raw = tmp_path / "explicit_raw.db"
    env_db = tmp_path / "env.db"
    env_raw = tmp_path / "env_raw.db"
    output_root = tmp_path / "out"

    monkeypatch.setenv("HOWL_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("HOWL_DB_PATH", str(env_db))
    monkeypatch.setenv("HOWL_RAW_DB_PATH", str(env_raw))

    env_paths = resolve_runtime_paths()
    explicit_paths = resolve_runtime_paths(db_path=explicit_db, raw_db_path=explicit_raw)

    assert env_paths.db_path == env_db
    assert env_paths.raw_db_path == env_raw
    assert explicit_paths.db_path == explicit_db
    assert explicit_paths.raw_db_path == explicit_raw
    assert explicit_paths.checkpoint_dir == output_root / "data" / "model_checkpoints"
