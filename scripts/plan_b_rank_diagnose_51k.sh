#!/usr/bin/env bash
#
# Skill-causal plan Step 2 — Gate B rank-use diagnostics on the 51k corpus.
#
# Probes the Plan B model for whether rank is under-used or over-leaked.
# Reads: data/model_checkpoints/plan_b_full_best.pt
# Writes: artifacts/rank_diagnosis.json (Gate B handoff)
#
# Usage:
#   scripts/plan_b_rank_diagnose_51k.sh [extra args forwarded to diagnose-rank]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACT="${REPO_ROOT}/artifacts/rank_diagnosis.json"
CKPT="${REPO_ROOT}/data/model_checkpoints/plan_b_full_best.pt"

mkdir -p "${REPO_ROOT}/artifacts"

cd "${REPO_ROOT}/code"

exec uv run python -m model.cli diagnose-rank \
  --checkpoint "${CKPT}" \
  --split train \
  --sample-games 2000 \
  --swap-sample 500 \
  --per-band-cap 500 \
  --max-anchors-per-layer 30000 \
  --artifact-path "${ARTIFACT}" \
  "$@"
