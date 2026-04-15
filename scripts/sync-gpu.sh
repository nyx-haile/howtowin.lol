#!/usr/bin/env bash
# sync-gpu.sh — bidirectional sync between alps (this laptop) and the WSL GPU host.
#
# Usage:
#   scripts/sync-gpu.sh push       # send corpus DBs + splits to GPU host
#   scripts/sync-gpu.sh pull        # fetch checkpoints back to laptop
#   scripts/sync-gpu.sh push-all    # push DBs, splits, AND secrets (rare)
#
# Prereqs:
#   - sshd running inside WSL on the GPU host (Microsoft docs: enable via
#     `sudo service ssh start`; persist with /etc/wsl.conf [boot] command)
#   - ssh-copy-id already run from this host to the WSL user@host below
#   - $HOWL_GPU_HOST env var set to the ssh target (e.g. nyx@xyn-himalayas.nord)
#     or pass it as: HOWL_GPU_HOST=nyx@xyn-himalayas.nord scripts/sync-gpu.sh push
#
# Paths assume the repo lives at the same relative location on both ends
# (~/build/howtowin.lol). Override with $HOWL_GPU_REPO.

set -euo pipefail

: "${HOWL_GPU_HOST:?set HOWL_GPU_HOST to user@host, e.g. nyx@xyn-himalayas.nord}"
HOWL_GPU_REPO="${HOWL_GPU_REPO:-~/build/howtowin.lol}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cmd="${1:-}"

case "$cmd" in
  push)
    rsync -avhP --mkpath \
      data/howtowin.db \
      data/raw_matches.db \
      data/splits/ \
      "$HOWL_GPU_HOST:$HOWL_GPU_REPO/data/"
    ;;
  pull)
    rsync -avhP --mkpath \
      "$HOWL_GPU_HOST:$HOWL_GPU_REPO/data/model_checkpoints/" \
      data/model_checkpoints/
    ;;
  push-all)
    rsync -avhP --mkpath \
      data/howtowin.db \
      data/raw_matches.db \
      data/splits/ \
      secrets/ \
      "$HOWL_GPU_HOST:$HOWL_GPU_REPO/data/"
    rsync -avhP --mkpath \
      secrets/ \
      "$HOWL_GPU_HOST:$HOWL_GPU_REPO/secrets/"
    ;;
  *)
    echo "usage: $0 {push|pull|push-all}" >&2
    exit 2
    ;;
esac
