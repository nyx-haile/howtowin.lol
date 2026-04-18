#!/usr/bin/env bash
# claude-state-sync.sh — sync per-project Claude Code state between machines
# via a private HF dataset repo.
#
# What gets synced:
#   ~/.claude/rules/                         (user global rules)
#   ~/.claude/settings.json                  (base settings)
#   ~/.claude/projects/<path-key>/memory/    (project auto-memory)
#
# The project dir on disk is path-encoded (/ and . -> -) and therefore
# differs between machines; this script re-derives it from the current
# repo root so `push` on one box and `pull` on another Just Work.
#
# Usage:
#   scripts/claude-state-sync.sh push    # local -> HF
#   scripts/claude-state-sync.sh pull    # HF -> local
#
# Env:
#   HOWL_STATE_REPO  (default: mer1yn/howl-agent-state)
#   HOWL_STATE_CANON (default: basename of repo root with . -> -)
#
# Prereqs: `hf` CLI or `uvx hf` logged in (`hf auth login`).

set -euo pipefail

REPO="${HOWL_STATE_REPO:-mer1yn/howl-agent-state}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANON="${HOWL_STATE_CANON:-$(basename "$REPO_ROOT" | tr . -)}"

# Resolve the `hf` invocation: prefer a standalone binary on PATH, fall
# back to `uvx hf` (how it's set up on the GPU host where we don't want
# to install hf into every venv).
if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v uvx >/dev/null 2>&1; then
  HF=(uvx hf)
else
  echo "error: neither 'hf' nor 'uvx' on PATH" >&2
  exit 1
fi

# Path-encode the repo root the same way Claude Code does: / and . both
# become -, leading - preserved.
PROJ_KEY="$(echo "$REPO_ROOT" | sed 's|/|-|g; s|\.|-|g')"
LOCAL_PROJ="$HOME/.claude/projects/$PROJ_KEY"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

say() { printf '[claude-state-sync] %s\n' "$*"; }

cmd="${1:-}"
case "$cmd" in
  push)
    say "repo=$REPO canon=$CANON local_proj=$LOCAL_PROJ"
    mkdir -p "$STAGE/claude/rules" "$STAGE/claude/projects/$CANON"
    [ -d "$HOME/.claude/rules" ] && cp -r "$HOME/.claude/rules/." "$STAGE/claude/rules/"
    [ -f "$HOME/.claude/settings.json" ] && cp "$HOME/.claude/settings.json" "$STAGE/claude/settings.json"
    if [ -d "$LOCAL_PROJ/memory" ]; then
      cp -r "$LOCAL_PROJ/memory" "$STAGE/claude/projects/$CANON/memory"
    else
      say "no memory dir at $LOCAL_PROJ/memory — skipping"
    fi
    say "staged contents:"
    (cd "$STAGE" && find . -type f | sort)
    "${HF[@]}" upload "$REPO" "$STAGE" --repo-type=dataset \
      --commit-message "sync $(date -u +%FT%TZ) from $(hostname)"
    ;;
  pull)
    say "repo=$REPO canon=$CANON local_proj=$LOCAL_PROJ"
    "${HF[@]}" download "$REPO" --repo-type=dataset --local-dir "$STAGE" >/dev/null
    mkdir -p "$HOME/.claude/rules" "$LOCAL_PROJ"
    [ -d "$STAGE/claude/rules" ] && rsync -a --delete "$STAGE/claude/rules/" "$HOME/.claude/rules/"
    [ -f "$STAGE/claude/settings.json" ] && cp "$STAGE/claude/settings.json" "$HOME/.claude/settings.json"
    if [ -d "$STAGE/claude/projects/$CANON/memory" ]; then
      mkdir -p "$LOCAL_PROJ/memory"
      rsync -a "$STAGE/claude/projects/$CANON/memory/" "$LOCAL_PROJ/memory/"
    fi
    ;;
  *)
    cat >&2 <<USAGE
usage: $0 {push|pull}
  push   local state  -> HF dataset ($REPO)
  pull   HF dataset   -> local state
USAGE
    exit 2
    ;;
esac

say "done: $cmd"
