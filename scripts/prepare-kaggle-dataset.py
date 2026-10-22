#!/usr/bin/env python3
"""Dry-run Kaggle dataset packaging for the current data snapshot.

This script never uploads anything. It validates the intended private Kaggle
Dataset contents, runs SQLite quick checks in read-only mode where feasible,
and writes a manifest bundle into a staging directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from runtime_paths import resolve_runtime_paths


TOTAL_SIZE_LIMIT_BYTES = 200 * 1024**3
DEFAULT_STAGING_DIR = REPO_ROOT / "artifacts" / "kaggle_dataset_manifest"
DEFAULT_PAYLOAD_DIR_NAME = "data"
ALLOWED_TOP_LEVEL = {"howtowin.db", "raw_matches.db", "splits"}
EXCLUDED_DIRECTORY_NAMES = {
    "artifacts",
    "cache",
    "checkpoints",
    "logs",
    "materialized_samples",
    "model_checkpoints",
    "reports",
    "retrieval",
    "__pycache__",
}
EXCLUDED_FILE_SUFFIXES = {
    ".log",
    ".tmp",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".db-wal",
    ".db-shm",
    ".sqlite-wal",
    ".sqlite-shm",
}


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


def is_sqlite_file(path: Path) -> bool:
    return path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def quick_check_sqlite(path: Path) -> str:
    uri = f"{path.expanduser().resolve().as_uri()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute("PRAGMA quick_check").fetchall()
    finally:
        conn.close()
    result = "; ".join(str(row[0]) for row in rows) if rows else ""
    if not result or result.lower() != "ok":
        fail(f"sqlite quick_check failed for {path}: {result or 'no result'}")
    return result


def _relative_parts(path: Path) -> tuple[str, ...]:
    return path.parts


def _excluded_reason(relative_path: Path) -> str | None:
    parts = _relative_parts(relative_path)
    if not parts:
        return None
    if parts[0] not in ALLOWED_TOP_LEVEL:
        return "not in Kaggle dataset allowlist"
    if len(parts) == 1:
        return None
    if parts[0] == "splits":
        return None
    if parts[0] in {"howtowin.db", "raw_matches.db"}:
        return "only the root database file is allowed"
    return None


def _prune_directory(name: str) -> bool:
    return name in EXCLUDED_DIRECTORY_NAMES


def _sidecar_paths(db_path: Path) -> list[Path]:
    return [db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")]


def collect_dataset_files(source_root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    included: list[Path] = []
    excluded: list[dict[str, str]] = []

    if not source_root.exists():
        fail(f"source root does not exist: {source_root}")

    for entry in sorted(source_root.iterdir(), key=lambda item: item.name):
        relative = entry.relative_to(source_root)
        if entry.is_dir():
            if _prune_directory(entry.name):
                for child in sorted(entry.rglob("*")):
                    if child.is_file():
                        excluded.append(
                            {"path": str(child.relative_to(source_root)), "reason": "excluded generated directory"}
                        )
                continue
            if entry.name != "splits":
                for child in sorted(entry.rglob("*")):
                    if child.is_file():
                        excluded.append(
                            {"path": str(child.relative_to(source_root)), "reason": "not in Kaggle dataset allowlist"}
                        )
                continue
            for child in sorted(entry.rglob("*")):
                if child.is_dir():
                    continue
                child_relative = child.relative_to(source_root)
                reason = _excluded_reason(child_relative)
                if reason is not None:
                    excluded.append({"path": str(child_relative), "reason": reason})
                    continue
                included.append(child)
            continue

        reason = _excluded_reason(relative)
        if reason is not None:
            excluded.append({"path": str(relative), "reason": reason})
            continue
        included.append(entry)

    return included, excluded


def validate_sidecars(included_files: list[Path]) -> None:
    for path in included_files:
        if not is_sqlite_file(path):
            continue
        for sidecar in _sidecar_paths(path):
            if sidecar.exists():
                fail(
                    f"hot SQLite sidecar detected for {path.name}: {sidecar.name}. "
                    "Close or checkpoint the database before packaging."
                )


def build_manifest(source_root: Path, staging_dir: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    staging_dir = staging_dir.resolve()
    included_files, excluded_entries = collect_dataset_files(source_root)
    validate_sidecars(included_files)

    included_entries: list[dict[str, object]] = []
    quick_checks: list[dict[str, str]] = []
    total_bytes = 0
    for path in included_files:
        relative = path.relative_to(source_root)
        size_bytes = file_size_bytes(path)
        total_bytes += size_bytes
        entry = {
            "path": str(relative),
            "upload_path": str(Path(DEFAULT_PAYLOAD_DIR_NAME) / relative),
            "size_bytes": size_bytes,
            "sha256": sha256_file(path),
        }
        included_entries.append(entry)
        if is_sqlite_file(path):
            quick_checks.append({"path": str(relative), "quick_check": quick_check_sqlite(path)})

    if total_bytes > TOTAL_SIZE_LIMIT_BYTES:
        fail(
            f"dataset size {total_bytes} bytes exceeds Kaggle limit {TOTAL_SIZE_LIMIT_BYTES} bytes"
        )

    manifest = {
        "status": "ok",
        "source_root": str(source_root),
        "staging_dir": str(staging_dir),
        "payload_dir": str(staging_dir / DEFAULT_PAYLOAD_DIR_NAME),
        "total_bytes": total_bytes,
        "size_limit_bytes": TOTAL_SIZE_LIMIT_BYTES,
        "included_file_count": len(included_entries),
        "excluded_file_count": len(excluded_entries),
        "included_files": included_entries,
        "excluded_files": excluded_entries,
        "sqlite_quick_check": quick_checks,
    }
    return manifest


def render_markdown(manifest: dict[str, object]) -> str:
    included = manifest["included_files"]
    excluded = manifest["excluded_files"]
    quick_checks = manifest["sqlite_quick_check"]
    lines = [
        "# Kaggle dataset dry-run manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Total bytes: `{manifest['total_bytes']}`",
        f"- Size limit bytes: `{manifest['size_limit_bytes']}`",
        f"- Included files: `{manifest['included_file_count']}`",
        f"- Excluded files: `{manifest['excluded_file_count']}`",
        "",
        "## Included files",
        "",
        "| Path | Size bytes | SHA256 |",
        "| --- | ---: | --- |",
    ]
    for entry in included:
        lines.append(f"| `{entry['path']}` | {entry['size_bytes']} | `{entry['sha256']}` |")
    lines.extend([
        "",
        "## Excluded files",
        "",
    ])
    for entry in excluded:
        lines.append(f"- `{entry['path']}` - {entry['reason']}")
    lines.extend([
        "",
        "## SQLite quick checks",
        "",
    ])
    for entry in quick_checks:
        lines.append(f"- `{entry['path']}` - {entry['quick_check']}")
    return "\n".join(lines) + "\n"


def render_readme(manifest: dict[str, object]) -> str:
    included_files = manifest["included_files"]
    return (
        "# howtowin Kaggle dataset snapshot\n\n"
        "Dry-run packaging output. This staging directory is upload-ready for a manual Kaggle Dataset upload; no Kaggle upload is performed.\n\n"
        f"- Source root: `{manifest['source_root']}`\n"
        f"- Payload directory: `{manifest['payload_dir']}`\n"
        f"- Total bytes: `{manifest['total_bytes']}`\n"
        f"- Size limit bytes: `{manifest['size_limit_bytes']}`\n"
        f"- Included files: `{manifest['included_file_count']}`\n"
        "\n"
        "## Included files\n\n"
        + "\n".join(f"- `{entry['upload_path']}`" for entry in included_files)
        + "\n"
    )


def render_checksums(manifest: dict[str, object]) -> str:
    lines = []
    for entry in manifest["included_files"]:
        lines.append(f"{entry['sha256']}  {entry['upload_path']}")
    return "\n".join(lines) + ("\n" if lines else "")


def _remove_existing_payload_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy_file(source: Path, dest: Path, strategy: str) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        _remove_existing_payload_path(dest)

    if strategy == "none":
        return "none"
    if strategy == "copy":
        shutil.copy2(source, dest)
        return "copy"
    if strategy == "symlink":
        os.symlink(source, dest)
        return "symlink"
    if strategy == "hardlink":
        os.link(source, dest)
        return "hardlink"
    if strategy == "auto":
        try:
            os.link(source, dest)
            return "hardlink"
        except OSError:
            os.symlink(source, dest)
            return "symlink"
    fail(f"unknown payload materialization strategy: {strategy}")


def materialize_payload(manifest: dict[str, object], strategy: str) -> dict[str, int]:
    source_root = Path(str(manifest["source_root"]))
    payload_dir = Path(str(manifest["payload_dir"]))
    counts: dict[str, int] = {"hardlink": 0, "symlink": 0, "copy": 0, "none": 0}

    for entry in manifest["included_files"]:
        source = source_root / str(entry["path"])
        dest = payload_dir / str(entry["path"])
        used_strategy = _link_or_copy_file(source, dest, strategy)
        counts[used_strategy] = counts.get(used_strategy, 0) + 1
    return counts


def write_manifest_bundle(manifest: dict[str, object], staging_dir: Path, payload_strategy: str = "auto") -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    materialized = materialize_payload(manifest, payload_strategy)
    manifest["payload_strategy"] = payload_strategy
    manifest["payload_materialized_counts"] = materialized
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging_dir / "manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    (staging_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    (staging_dir / "checksums.txt").write_text(render_checksums(manifest), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    runtime_paths = resolve_runtime_paths()
    p = argparse.ArgumentParser(description="Dry-run the Kaggle dataset packaging manifest.")
    p.add_argument(
        "--source-root",
        type=Path,
        default=runtime_paths.data_root,
        help="Source data root to scan. Defaults to the resolved local data root.",
    )
    p.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Directory where manifest JSON/Markdown and checksum files are written.",
    )
    p.add_argument(
        "--payload-strategy",
        choices=("auto", "hardlink", "symlink", "copy", "none"),
        default="auto",
        help="How to place allowed payload under staging/data. auto tries hardlinks then symlinks; copy is explicit.",
    )
    p.add_argument(
        "--print-manifest",
        action="store_true",
        help="Print the JSON manifest to stdout after writing the staging bundle.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest = build_manifest(args.source_root, args.staging_dir)
    write_manifest_bundle(manifest, args.staging_dir, payload_strategy=args.payload_strategy)
    if args.print_manifest:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(render_markdown(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
