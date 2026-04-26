#!/usr/bin/env bash
#
# Skill-causal plan Step 4 — Gate D counterfactual intervention scan on the
# 51k corpus.
#
# Reads:
#   data/model_checkpoints/plan_b_full_best.pt
# Writes:
#   artifacts/intervention_candidates.json    (Gate D handoff)
#
# Defaults to a 500-game sample of the game-cold split (~3 min on CPU);
# override --max-games for deeper sweeps. CPU is the default device because
# the GTX 1650's 4 GB VRAM cannot fit a per-game forward pass plus the
# intervention rollouts.
#
# Usage:
#   scripts/plan_b_intervention_scan_51k.sh [extra args forwarded to intervention-scan]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${REPO_ROOT}/data/model_checkpoints/plan_b_full_best.pt"
ARTIFACT="${REPO_ROOT}/artifacts/intervention_candidates.json"

mkdir -p "${REPO_ROOT}/artifacts"

cd "${REPO_ROOT}/code"

: "${HOWL_EVAL_DEVICE:=cpu}"
: "${HOWL_INTERVENTION_MAX_GAMES:=500}"
: "${HOWL_INTERVENTION_MAX_ANCHORS:=4}"
: "${HOWL_INTERVENTION_N_STEPS:=3}"

exec uv run python -m model.cli intervention-scan \
  --checkpoint "${CKPT}" \
  --artifact-path "${ARTIFACT}" \
  --device "${HOWL_EVAL_DEVICE}" \
  --split game_cold \
  --max-games "${HOWL_INTERVENTION_MAX_GAMES}" \
  --max-anchors-per-game "${HOWL_INTERVENTION_MAX_ANCHORS}" \
  --n-steps "${HOWL_INTERVENTION_N_STEPS}" \
  "$@"
