# Kaggle Training Workflow

This runbook is for a private, version-pinned Kaggle Dataset mounted into a Kaggle Notebook. It keeps SQLite inputs read-only under `/kaggle/input` and writes every generated file under `/kaggle/working/howtowin`.

Do not copy the 64 GB `howtowin.db` into `/kaggle/working`. Kaggle mounts the Dataset as input storage. Use `/kaggle/working` only for checkpoints, reports, smoke artifacts, and files you want saved with the notebook output.

Kaggle is a new snapshot workflow alongside the existing WSL GPU rsync path in `scripts/sync-gpu.sh`. Use `sync-gpu.sh` for mutable remote-host iteration. Use Kaggle when you want a free GPU run against the current private Dataset snapshot. This is not the future live mutable corpus, streaming ingest path, or 100x data lake design.

## Workflow Choice

| Need | Use | Why |
|---|---|---|
| Local development or mutable remote-host iteration | `scripts/sync-gpu.sh` | It pushes DBs and splits to the WSL GPU host over SSH and pulls checkpoints back. |
| Free GPU training from a fixed corpus snapshot | Kaggle private Dataset plus Notebook | Kaggle mounts the Dataset read-only and saves selected outputs from `/kaggle/working`. |
| Future live mutable corpus or 100x data lake | Neither this runbook nor Kaggle Datasets | That architecture needs a different storage and ingest design. |

## Private Dataset Packaging And Manual Upload

The repo provides a dry-run packaging helper. It stages the allowed payload plus metadata and checks the intended contents, but it does not upload data or call the Kaggle API.

Run this locally from the repo root:

```bash
uv run python scripts/prepare-kaggle-dataset.py \
  --source-root data \
  --staging-dir /tmp/howtowin-kaggle-staging
```

Review the generated staging files before any upload:

- `data/howtowin.db`, `data/raw_matches.db`, and `data/splits/` should exist under the staging directory. By default the helper uses hardlinks where possible and falls back to symlinks so it does not duplicate the large DB files; use `--payload-strategy copy` only when an operator intentionally wants physical copies.
- `manifest.md`, `manifest.json`, `checksums.txt`, and `README.md` should sit at the staging root and describe the staged payload.
- Generated checkpoints, logs, reports, retrieval indexes, retrieval caches, and materialized samples should be excluded.
- SQLite WAL or SHM sidecars should be resolved before packaging so the uploaded DB snapshot is clean.

If the manifest is correct, an operator may run Kaggle CLI commands manually from a machine that already has Kaggle credentials configured outside this repo. These commands are examples only. They are not automated acceptance steps, and this document does not claim that an upload has succeeded.

```bash
kaggle datasets init -p /tmp/howtowin-kaggle-staging
kaggle datasets create -p /tmp/howtowin-kaggle-staging --private
kaggle datasets version -p /tmp/howtowin-kaggle-staging -m "refresh current training snapshot"
```

Never commit `kaggle.json`, `KAGGLE_KEY`, OAuth material, tokens, account-specific config, or notebook secrets. Keep dataset slugs as placeholders in docs unless an operator records them in a private run log outside git.

## Dataset Placeholder

Replace both placeholders before running:

```text
<dataset-slug> = <owner>/<dataset-name>
<dataset-version> = <integer-version-id>
```

Pin the Dataset version in the notebook data source panel or notebook metadata. Do not rely silently on `latest`, because a later Dataset version can change the DB snapshot, split files, or checksums under the same slug.

Expected mounted files:

```text
/kaggle/input/<dataset-slug>/data/howtowin.db
/kaggle/input/<dataset-slug>/data/raw_matches.db
/kaggle/input/<dataset-slug>/data/splits/plan_a_holdout.txt
/kaggle/input/<dataset-slug>/data/splits/plan_b_cold_holdout.txt
```

## Kaggle Notebook Setup

1. Attach the private Dataset at the pinned `<dataset-slug>` and `<dataset-version>`.
2. In notebook settings, enable GPU if you want a CUDA check. Kaggle free GPU availability can queue, and P100 availability is a caveat, not a guarantee.
3. Turn internet on only if you need dependency setup from the repo or package index. If the notebook image already has the repo and dependencies, leave internet off.
4. Do not create or upload `kaggle.json`, tokens, OAuth material, or other credentials in the notebook.
5. Keep the 12 hour CPU/GPU session cap and the 20 GB `/kaggle/working` persisted output cap in mind. Remove unneeded checkpoints before saving a notebook version.

Keep checkpoints, caches, and reports below 20 GB of saved `/kaggle/working` output. Prefer one best checkpoint plus the small smoke/preflight reports. Delete intermediate checkpoints, cache files, and bulky reports before clicking Save Version.

## Environment Cell

Paste this into the first Kaggle cell after replacing `<dataset-slug>`:

```python
import os

DATASET_SLUG = "<dataset-slug>"
DATASET_VERSION = "<dataset-version>"

os.environ["HOWL_DB_PATH"] = f"file:/kaggle/input/{DATASET_SLUG}/data/howtowin.db?mode=ro&immutable=1"
os.environ["HOWL_RAW_DB_PATH"] = f"file:/kaggle/input/{DATASET_SLUG}/data/raw_matches.db?mode=ro&immutable=1"
os.environ["HOWL_OUTPUT_ROOT"] = "/kaggle/working/howtowin"
os.environ["HOWL_INPUT_ROOT"] = f"/kaggle/input/{DATASET_SLUG}"

print("Pinned Kaggle Dataset", DATASET_SLUG, "version", DATASET_VERSION)
print("HOWL_DB_PATH=", os.environ["HOWL_DB_PATH"])
print("HOWL_RAW_DB_PATH=", os.environ["HOWL_RAW_DB_PATH"])
print("HOWL_OUTPUT_ROOT=", os.environ["HOWL_OUTPUT_ROOT"])
```

Exact shell form, useful for a script cell:

```bash
export HOWL_DB_PATH='file:/kaggle/input/<dataset-slug>/data/howtowin.db?mode=ro&immutable=1'
export HOWL_RAW_DB_PATH='file:/kaggle/input/<dataset-slug>/data/raw_matches.db?mode=ro&immutable=1'
export HOWL_OUTPUT_ROOT='/kaggle/working/howtowin'
export HOWL_INPUT_ROOT='/kaggle/input/<dataset-slug>'
```

## Dependency Cell

Use the project toolchain. If the repo is already present in the notebook image, run from the repo root:

```bash
uv sync
```

If `uv` is missing and internet is enabled, install it in the notebook session first. Do not commit notebook credentials or local provider config back to the repo.

## Bounded Training Smoke

Use the bounded smoke script below as the two-step Kaggle smoke gate. It checks imports and parser wiring, opens both SQLite DBs through the read-only helpers with `mode=ro` and `immutable=1`, runs `PRAGMA quick_check`, then invokes `plan-b-train` through the shared training loop with `--max-steps 2` semantics. It writes one small JSON file under `/kaggle/working/howtowin`.

```bash
uv run python scripts/kaggle_smoke.py --steps 2
```

Expected result:

```text
"status": "ok"
"step": "imports"
"step": "training-smoke"
"requested_optimizer_steps": 2
"artifact": "/kaggle/working/howtowin/kaggle_smoke_result.json"
```

For a local fake-Kaggle fixture, pass `--allow-local-paths --input-root <fake-input-root>`. Do not use that flag in a real Kaggle notebook.

## Training Preflight

After the smoke gate passes, run the existing training preflight before any full training attempt:

```bash
uv run python -m model.cli plan-b-train \
  --preflight-only \
  --no-preflight-strict \
  --num-workers 0 \
  --train-cache-size 0 \
  --materialized-cache-mode off
```

This writes the preflight artifact under `/kaggle/working/howtowin/artifacts/`. It should not write under `/kaggle/input`.

## Full Training Start Point

Only start a full run after the smoke and preflight pass:

```bash
uv run python -m model.cli plan-b-train \
  --epochs 30 \
  --batch-size 8 \
  --train-cache-size 0 \
  --materialized-cache-mode off
```

Watch `/kaggle/working/howtowin/data/model_checkpoints/`, `/kaggle/working/howtowin/data/retrieval/`, cache directories, and `/kaggle/working/howtowin/artifacts/` so saved outputs stay under the 20 GB cap. If the GPU session is queued, interrupted, or reaches the 12 hour limit, save only the needed checkpoint and artifact files.

Before saving a notebook version, clean up anything not needed for handoff:

```bash
du -sh /kaggle/working/howtowin/*
rm -rf /kaggle/working/howtowin/data/retrieval/cache
rm -rf /kaggle/working/howtowin/data/materialized_samples
```

Do not delete the final checkpoint or reports you intend to preserve. The goal is to keep checkpoints, caches, and reports together below 20 GB of saved output.
