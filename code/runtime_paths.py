"""Shared runtime path resolution for local and Kaggle-style execution."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HOWL_INPUT_ROOT_ENV = "HOWL_INPUT_ROOT"
HOWL_OUTPUT_ROOT_ENV = "HOWL_OUTPUT_ROOT"
HOWL_DB_PATH_ENV = "HOWL_DB_PATH"
HOWL_RAW_DB_PATH_ENV = "HOWL_RAW_DB_PATH"

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RuntimePaths:
    input_root: Path
    output_root: Path
    data_root: Path
    db_path: Path
    raw_db_path: Path
    split_dir: Path
    checkpoint_dir: Path
    retrieval_dir: Path
    cache_root: Path
    report_artifact_root: Path
    materialized_sample_root: Path


def _path_from(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return _path_from(raw)


def resolve_runtime_paths(
    *,
    input_root: str | os.PathLike[str] | None = None,
    output_root: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
    raw_db_path: str | os.PathLike[str] | None = None,
) -> RuntimePaths:
    resolved_input_root = _path_from(input_root) or _env_path(HOWL_INPUT_ROOT_ENV) or REPO_ROOT
    resolved_output_root = _path_from(output_root) or _env_path(HOWL_OUTPUT_ROOT_ENV) or REPO_ROOT
    data_root = resolved_input_root / "data"
    output_data_root = resolved_output_root / "data"

    resolved_db_path = (
        _path_from(db_path)
        or _env_path(HOWL_DB_PATH_ENV)
        or data_root / "howtowin.db"
    )
    resolved_raw_db_path = (
        _path_from(raw_db_path)
        or _env_path(HOWL_RAW_DB_PATH_ENV)
        or data_root / "raw_matches.db"
    )

    return RuntimePaths(
        input_root=resolved_input_root,
        output_root=resolved_output_root,
        data_root=data_root,
        db_path=resolved_db_path,
        raw_db_path=resolved_raw_db_path,
        split_dir=data_root / "splits",
        checkpoint_dir=output_data_root / "model_checkpoints",
        retrieval_dir=output_data_root / "retrieval",
        cache_root=output_data_root / "retrieval" / "cache",
        report_artifact_root=resolved_output_root / "artifacts",
        materialized_sample_root=output_data_root / "materialized_samples" / "plan_b",
    )


def artifact_path(name: str) -> Path:
    return resolve_runtime_paths().report_artifact_root / name
