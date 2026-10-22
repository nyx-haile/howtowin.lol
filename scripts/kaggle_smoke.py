#!/usr/bin/env python3
"""Kaggle and local smoke checks for the read-only SQLite workflow."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _path_is_under(child: Path, parent: Path) -> bool:
    child = child.resolve()
    parent = parent.resolve()
    try:
        return child == parent or child.is_relative_to(parent)
    except AttributeError:
        return os.path.commonpath([str(child), str(parent)]) == str(parent)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


def _quick_check(conn, label: str) -> str:
    result = conn.execute("PRAGMA quick_check").fetchone()[0]
    if result != "ok":
        raise RuntimeError(f"{label} quick_check failed: {result}")
    return result


def _run_import_step() -> dict[str, object]:
    import db
    import raw_db
    from model import cli
    from runtime_paths import resolve_runtime_paths

    paths = resolve_runtime_paths()
    parser = cli.build_parser()
    parser.parse_args(["plan-b-train", "--preflight-only"])
    return {
        "step": "imports",
        "status": "ok",
        "db_helper": db.get_conn.__name__,
        "raw_db_helper": raw_db.get_raw_conn.__name__,
        "output_root": str(paths.output_root),
    }


def _run_db_step(args: argparse.Namespace) -> dict[str, object]:
    import db
    import raw_db
    from runtime_paths import resolve_runtime_paths

    db_uri = _require_env("HOWL_DB_PATH")
    raw_db_uri = _require_env("HOWL_RAW_DB_PATH")
    _require_env("HOWL_OUTPUT_ROOT")

    paths = resolve_runtime_paths()
    output_root = paths.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.allow_local_paths:
        expected_input = Path(args.expected_input_root)
        expected_output = Path(args.expected_output_root)
        if "/kaggle/input/" not in db_uri or "/kaggle/input/" not in raw_db_uri:
            raise SystemExit("HOWL_DB_PATH and HOWL_RAW_DB_PATH must point at /kaggle/input on Kaggle")
        if not _path_is_under(output_root, expected_output):
            raise SystemExit(f"HOWL_OUTPUT_ROOT must be under {expected_output}")
        if str(expected_input) in str(output_root):
            raise SystemExit("HOWL_OUTPUT_ROOT must not point into /kaggle/input")

    input_root_text = args.input_root or ""
    if input_root_text:
        input_root = Path(input_root_text).expanduser()
        if _path_is_under(output_root, input_root):
            raise SystemExit("output root must not be under the input root")

    main_conn = db.get_conn(db_uri, readonly=True)
    try:
        main_query_only = int(main_conn.execute("PRAGMA query_only").fetchone()[0])
        main_quick_check = _quick_check(main_conn, "main DB")
        game_count = int(main_conn.execute("SELECT COUNT(*) FROM games").fetchone()[0])
    finally:
        main_conn.close()

    raw_conn = raw_db.get_raw_conn(raw_db_uri, readonly=True)
    try:
        raw_query_only = int(raw_conn.execute("PRAGMA query_only").fetchone()[0])
        raw_quick_check = _quick_check(raw_conn, "raw DB")
        raw_match_count = int(raw_conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0])
    finally:
        raw_conn.close()

    if main_query_only != 1 or raw_query_only != 1:
        raise RuntimeError("read-only helpers did not enable PRAGMA query_only")

    artifact = output_root / args.artifact_name
    if not _path_is_under(artifact, output_root):
        raise SystemExit("artifact path escaped HOWL_OUTPUT_ROOT")

    result = {
        "step": "db-connectivity",
        "status": "ok",
        "game_count": game_count,
        "raw_match_count": raw_match_count,
        "main_quick_check": main_quick_check,
        "raw_quick_check": raw_quick_check,
        "artifact": str(artifact),
    }
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _run_training_step(args: argparse.Namespace) -> dict[str, object]:
    from model.dataset import load_split
    from model.plan_b_train import plan_b_train_loop
    from runtime_paths import resolve_runtime_paths

    connectivity = _run_db_step(args)
    train_ids = load_split("train")[: args.max_train_matches]
    val_ids = load_split("holdout")[: args.max_eval_matches]
    cold_ids = load_split("cold")[: args.max_eval_matches]
    if not train_ids:
        raise SystemExit("no train matches available for Kaggle smoke training")

    training_steps = args.training_steps if args.training_steps is not None else args.steps
    with contextlib.redirect_stdout(sys.stderr):
        history = plan_b_train_loop(
            train_ids,
            val_ids,
            cold_ids,
            epochs=1,
            batch_size=args.batch_size,
            lr=args.lr,
            max_puuids=args.max_puuids,
            log_every=1,
            checkpoint_tag="kaggle_smoke",
            num_workers=0,
            train_cache_size=0,
            run_preflight=False,
            materialized_cache_mode="off",
            compile_model=False,
            max_steps=training_steps,
        )

    output_root = resolve_runtime_paths().output_root.expanduser()
    artifact = output_root / args.artifact_name
    if not _path_is_under(artifact, output_root):
        raise SystemExit("artifact path escaped HOWL_OUTPUT_ROOT")

    result = {
        "step": "training-smoke",
        "status": "ok",
        "requested_optimizer_steps": training_steps,
        "train_match_count": len(train_ids),
        "val_match_count": len(val_ids),
        "cold_match_count": len(cold_ids),
        "train_loss_count": len(history["train_loss"]),
        "db_connectivity": connectivity,
        "artifact": str(artifact),
    }
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps < 1 or args.steps > 2:
        raise SystemExit("--steps must be 1 or 2")

    completed = [_run_import_step()]
    if args.steps >= 2:
        completed.append(_run_training_step(args))
    summary = {"status": "ok", "steps": completed}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2, help="Run one import step or both smoke steps.")
    parser.add_argument(
        "--artifact-name",
        default="kaggle_smoke_result.json",
        help="Result JSON filename written under HOWL_OUTPUT_ROOT.",
    )
    parser.add_argument(
        "--allow-local-paths",
        action="store_true",
        help="Allow fake Kaggle paths under a local temp directory for tests.",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Optional fake input root used to assert outputs are separate during local tests.",
    )
    parser.add_argument("--expected-input-root", default="/kaggle/input")
    parser.add_argument("--expected-output-root", default="/kaggle/working")
    parser.add_argument("--training-steps", type=int, default=None, help="Optimizer steps for the bounded training smoke. Defaults to --steps.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-puuids", type=int, default=64)
    parser.add_argument("--max-train-matches", type=int, default=2)
    parser.add_argument("--max-eval-matches", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
