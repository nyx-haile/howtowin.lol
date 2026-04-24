#!/usr/bin/env bash
#
# Skill-causal plan Step 1 — Gate A preflight on the full 51k-game corpus.
#
# Runs the Plan B runtime preflight targeted at the remote compute server
# (GTX 1650 SUPER / 4GB VRAM / 16GB RAM / Ryzen 5 / 12 threads) with the
# <=24h early-stop budget and materialized-sample cache enabled.
#
# Writes: artifacts/runtime_preflight.json  (Gate A handoff)
#
# Usage:
#   scripts/plan_b_preflight_51k.sh [extra args forwarded to plan-b-train]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACT="${REPO_ROOT}/artifacts/runtime_preflight.json"
CACHE_DIR="${REPO_ROOT}/data/materialized_samples/plan_b"

mkdir -p "${REPO_ROOT}/artifacts"

cd "${REPO_ROOT}/code"

exec uv run python -m model.cli plan-b-train \
  --preflight-only \
  --no-preflight-strict \
  --preflight-artifact "${ARTIFACT}" \
  --preflight-max-earlystop-hours 24.0 \
  --preflight-max-epoch-minutes 180.0 \
  --materialized-cache-dir "${CACHE_DIR}" \
  --materialized-cache-mode readwrite \
  --materialized-cache-version v1 \
  --no-materialized-cache-warmup \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 12 \
  --no-compile \
  "$@"
