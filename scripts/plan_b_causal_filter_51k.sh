#!/usr/bin/env bash
#
# Skill-causal plan Step 5 — Gate E causal validation filter on the 51k corpus.
#
# Reads:
#   data/model_checkpoints/plan_b_full_best.pt
#   artifacts/intervention_candidates.json   (Step 4 output)
# Writes:
#   artifacts/causal_filter_report.json      (Gate E handoff)
#
# CPU is the default device — sklearn estimators run on numpy arrays. The
# model forward pass is small enough that GTX 1650 (4 GB) could host it,
# but we keep the whole pipeline on CPU to avoid copying tensors back per
# game.
#
# Usage:
#   scripts/plan_b_causal_filter_51k.sh [extra args forwarded to causal-filter]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${REPO_ROOT}/data/model_checkpoints/plan_b_full_best.pt"
CAND="${REPO_ROOT}/artifacts/intervention_candidates.json"
ARTIFACT="${REPO_ROOT}/artifacts/causal_filter_report.json"

mkdir -p "${REPO_ROOT}/artifacts"

cd "${REPO_ROOT}/code"

: "${HOWL_EVAL_DEVICE:=cpu}"
: "${HOWL_CAUSAL_MAX_GAMES:=500}"

exec uv run python -m model.cli causal-filter \
  --checkpoint "${CKPT}" \
  --candidate-path "${CAND}" \
  --artifact-path "${ARTIFACT}" \
  --device "${HOWL_EVAL_DEVICE}" \
  --split game_cold \
  --max-games "${HOWL_CAUSAL_MAX_GAMES}" \
  "$@"
