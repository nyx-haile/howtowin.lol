#!/usr/bin/env bash
#
# Skill-causal plan Step 3 — Gate C skill-aware retrieval eval on the 51k corpus.
#
# Reads:
#   data/model_checkpoints/plan_b_full_best.pt
#   data/retrieval/plan_b_index.pt        (rebuilt first if schema < 2)
# Writes:
#   artifacts/retrieval_eval.json         (Gate C handoff)
#
# Usage:
#   scripts/plan_b_skill_retrieval_eval_51k.sh [extra args forwarded to skill-retrieval-eval]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${REPO_ROOT}/data/model_checkpoints/plan_b_full_best.pt"
INDEX="${REPO_ROOT}/data/retrieval/plan_b_index.pt"
ARTIFACT="${REPO_ROOT}/artifacts/retrieval_eval.json"

mkdir -p "${REPO_ROOT}/artifacts"

cd "${REPO_ROOT}/code"

# Detect whether the current index has rank metadata (schema v2+). If not,
# rebuild before eval — skill-aware retrieval requires it.
SCHEMA=$(uv run python -c "
from model.retrieval import load_index
try:
    b = load_index('${INDEX}')
    print(b.schema_version if b.has_rank_metadata() else 1)
except FileNotFoundError:
    print(0)
" 2>/dev/null || echo "0")

if [ "${SCHEMA}" -lt 2 ]; then
    echo "[skill-retrieval-eval] index schema=${SCHEMA} — rebuilding with rank-band metadata..."
    uv run python -m model.cli retrieval-build
fi

: "${HOWL_EVAL_DEVICE:=cpu}"

exec uv run python -m model.cli skill-retrieval-eval \
  --checkpoint "${CKPT}" \
  --index-path "${INDEX}" \
  --artifact-path "${ARTIFACT}" \
  --k 64 \
  --min-effective-k 32 \
  --out-of-band-penalty 1.25 \
  --auc-minute 15 \
  --splits game_cold,player_cold \
  --device "${HOWL_EVAL_DEVICE}" \
  "$@"
