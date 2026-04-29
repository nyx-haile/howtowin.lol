"""Disk cache for M4 retrieval-eval query keys and cohort indices.

Cache key: SHA256 over:
  model / static_only:  checkpoint_sha + sorted(holdout_match_ids) + source_hash
  frame_features:       sorted(holdout_match_ids) + source_hash  (checkpoint-independent)

Source modules hashed (raw bytes):
  model / static_only:  encoders.py, rssm.py, plan_b_model.py, dataset.py, retrieval.py
  frame_features:       baselines/frame_features_index.py

Layout under data/retrieval/cache/:
  {kind}_{holdout_label}_{key16}.pt           — {keys, minutes} tensors
  {kind}_{holdout_label}_{key16}.manifest.json — provenance sidecar
  {kind}_{holdout_label}_{key16}_cohort_idx.pt — cohort_idx tensor
"""
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch

_CACHE_DIR = (Path(os.path.dirname(__file__)) / ".." / ".." / "data" / "retrieval" / "cache").resolve()

_MODEL_MODULES = ["encoders.py", "rssm.py", "plan_b_model.py", "dataset.py", "retrieval.py"]
_FF_MODULES = [os.path.join("baselines", "frame_features_index.py")]

_SOURCE_MODULES: dict[str, list[str]] = {
    "model": _MODEL_MODULES,
    "static_only": _MODEL_MODULES,
    "frame_features": _FF_MODULES,
}
_USE_CHECKPOINT_SHA: dict[str, bool] = {
    "model": True,
    "static_only": True,
    "frame_features": False,
}


def compute_source_hash(kind: str) -> str:
    """SHA256 over the raw bytes of the source modules for the given eval kind."""
    pkg_dir = Path(os.path.dirname(__file__))
    modules = _SOURCE_MODULES.get(kind, _MODEL_MODULES)
    h = hashlib.sha256()
    for name in sorted(modules):
        p = pkg_dir / name
        try:
            h.update(p.read_bytes())
        except FileNotFoundError:
            h.update(f"MISSING:{name}".encode())
    return h.hexdigest()


def get_cache_key(kind: str, *, checkpoint_sha: str | None, holdout_match_ids: list[str]) -> str:
    """Deterministic cache key for the given eval kind and holdout membership."""
    source_hash = compute_source_hash(kind)
    parts: list[str] = []
    if _USE_CHECKPOINT_SHA.get(kind, True) and checkpoint_sha:
        parts.append(checkpoint_sha)
    parts.append("\n".join(sorted(holdout_match_ids)))
    parts.append(source_hash)
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _stem(kind: str, holdout_label: str, cache_key: str) -> str:
    return f"{kind}_{holdout_label}_{cache_key[:16]}"


def _keys_path(kind: str, holdout_label: str, cache_key: str) -> Path:
    return _CACHE_DIR / f"{_stem(kind, holdout_label, cache_key)}.pt"


def _manifest_path(kind: str, holdout_label: str, cache_key: str) -> Path:
    return _CACHE_DIR / f"{_stem(kind, holdout_label, cache_key)}.manifest.json"


def _cohort_idx_path(kind: str, holdout_label: str, cache_key: str) -> Path:
    return _CACHE_DIR / f"{_stem(kind, holdout_label, cache_key)}_cohort_idx.pt"


def _git_head_sha() -> str:
    try:
        root = str((Path(os.path.dirname(__file__)) / ".." / "..").resolve())
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _atomic_save(path: Path, payload: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(_CACHE_DIR), prefix=".evcache.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_cached_keys(
    kind: str,
    holdout_label: str,
    *,
    checkpoint_sha: str | None,
    holdout_match_ids: list[str],
) -> dict | None:
    """Return {keys, minutes} tensors from cache, or None on miss.

    Prints manifest provenance on every hit so stale caches are visible.
    """
    cache_key = get_cache_key(kind, checkpoint_sha=checkpoint_sha, holdout_match_ids=holdout_match_ids)
    p = _keys_path(kind, holdout_label, cache_key)
    if not p.exists():
        return None
    payload = torch.load(str(p), map_location="cpu", weights_only=True)
    mp = _manifest_path(kind, holdout_label, cache_key)
    if mp.exists():
        try:
            m = json.loads(mp.read_text())
            print(
                f"[eval_cache] HIT {kind}/{holdout_label}: "
                f"n_keys={m.get('n_keys')} key_dim={m.get('key_dim')} "
                f"checkpoint={m.get('checkpoint_sha','?')} "
                f"code={m.get('code_sha','?')} "
                f"encoded_at={m.get('encoded_at','?')}",
                flush=True,
            )
        except Exception:
            print(f"[eval_cache] HIT {kind}/{holdout_label}: (manifest unreadable)", flush=True)
    else:
        print(f"[eval_cache] HIT {kind}/{holdout_label}: (no manifest sidecar)", flush=True)
    return payload


def save_cached_keys(
    kind: str,
    holdout_label: str,
    *,
    checkpoint_sha: str | None,
    holdout_match_ids: list[str],
    keys: torch.Tensor,
    minutes: torch.Tensor,
) -> None:
    """Atomically persist (keys, minutes) with a manifest sidecar."""
    cache_key = get_cache_key(kind, checkpoint_sha=checkpoint_sha, holdout_match_ids=holdout_match_ids)
    p = _keys_path(kind, holdout_label, cache_key)
    _atomic_save(p, {"keys": keys.cpu(), "minutes": minutes.cpu()})
    manifest = {
        "kind": kind,
        "holdout_label": holdout_label,
        "checkpoint_sha": checkpoint_sha or "n/a",
        "code_sha": _git_head_sha(),
        "source_hash": compute_source_hash(kind),
        "n_keys": int(keys.shape[0]),
        "key_dim": int(keys.shape[1]) if keys.ndim == 2 else -1,
        "encoded_at": int(time.time()),
    }
    mp = _manifest_path(kind, holdout_label, cache_key)
    fd, tmp = tempfile.mkstemp(dir=str(_CACHE_DIR), prefix=".evcache.", suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, str(mp))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(
        f"[eval_cache] SAVED {kind}/{holdout_label}: "
        f"n_keys={manifest['n_keys']} → {p.name}",
        flush=True,
    )


def load_cached_cohort_idx(
    kind: str,
    holdout_label: str,
    *,
    checkpoint_sha: str | None,
    holdout_match_ids: list[str],
) -> torch.Tensor | None:
    """Return cached cohort_idx tensor (Q, k) or None on miss."""
    cache_key = get_cache_key(kind, checkpoint_sha=checkpoint_sha, holdout_match_ids=holdout_match_ids)
    p = _cohort_idx_path(kind, holdout_label, cache_key)
    if not p.exists():
        return None
    payload = torch.load(str(p), map_location="cpu", weights_only=True)
    t = payload["cohort_idx"]
    print(
        f"[eval_cache] HIT cohort_idx {kind}/{holdout_label}: shape={tuple(t.shape)}",
        flush=True,
    )
    return t


def save_cached_cohort_idx(
    kind: str,
    holdout_label: str,
    *,
    checkpoint_sha: str | None,
    holdout_match_ids: list[str],
    cohort_idx: torch.Tensor,
) -> None:
    """Atomically persist cohort_idx tensor."""
    cache_key = get_cache_key(kind, checkpoint_sha=checkpoint_sha, holdout_match_ids=holdout_match_ids)
    p = _cohort_idx_path(kind, holdout_label, cache_key)
    _atomic_save(p, {"cohort_idx": cohort_idx.cpu()})
    print(
        f"[eval_cache] SAVED cohort_idx {kind}/{holdout_label}: "
        f"shape={tuple(cohort_idx.shape)} → {p.name}",
        flush=True,
    )


def delete_cached(
    kind: str,
    holdout_label: str,
    *,
    checkpoint_sha: str | None,
    holdout_match_ids: list[str],
) -> int:
    """Delete all cache files for this (kind, holdout_label, key). Returns count deleted."""
    cache_key = get_cache_key(kind, checkpoint_sha=checkpoint_sha, holdout_match_ids=holdout_match_ids)
    targets = [
        _keys_path(kind, holdout_label, cache_key),
        _manifest_path(kind, holdout_label, cache_key),
        _cohort_idx_path(kind, holdout_label, cache_key),
    ]
    deleted = 0
    for p in targets:
        if p.exists():
            p.unlink()
            deleted += 1
    return deleted
