"""Reusable on-disk materialization for Plan B training samples."""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

DEFAULT_SAMPLE_CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "materialized_samples",
    "plan_b",
)
SAMPLE_CACHE_DIR_ENV = "HOWL_PLANB_SAMPLE_CACHE_DIR"
SAMPLE_CACHE_MODE_ENV = "HOWL_PLANB_SAMPLE_CACHE_MODE"
SAMPLE_CACHE_VERSION_ENV = "HOWL_PLANB_SAMPLE_CACHE_VERSION"
SAMPLE_CACHE_WARMUP_ENV = "HOWL_PLANB_SAMPLE_CACHE_WARMUP"
DEFAULT_SAMPLE_CACHE_VERSION = "v1"

_READ_MODES = {"readonly", "readwrite"}
_WRITE_MODES = {"readwrite", "refresh"}


def _flag_from_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _stable_hash_bytes(parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _puuid_index_signature(puuid_index: Mapping[str, int]) -> str:
    if not puuid_index:
        return "none"
    parts = [
        f"{puuid}:{idx}".encode("utf-8")
        for puuid, idx in sorted(puuid_index.items())
    ]
    return _stable_hash_bytes(parts)


def _exclude_signature(exclude_match_ids: Sequence[str] | set[str] | None) -> str:
    if not exclude_match_ids:
        return "none"
    return _stable_hash_bytes([match_id.encode("utf-8") for match_id in sorted(exclude_match_ids)])


def _normalize_mode(mode: str | None, cache_dir: str | None) -> str:
    if mode is None:
        env_mode = os.environ.get(SAMPLE_CACHE_MODE_ENV)
        if env_mode:
            mode = env_mode
        elif cache_dir or os.environ.get(SAMPLE_CACHE_DIR_ENV):
            mode = "readwrite"
        else:
            mode = "off"
    normalized = str(mode).strip().lower()
    if normalized in {"off", "disabled", "none", ""}:
        return "off"
    if normalized in {"readwrite", "rw", "write", "read-write"}:
        return "readwrite"
    if normalized in {"readonly", "ro", "read-only"}:
        return "readonly"
    if normalized in {"refresh", "rebuild", "overwrite"}:
        return "refresh"
    raise ValueError(f"unknown materialized sample cache mode: {mode!r}")


@dataclass(frozen=True)
class MaterializedSampleCacheConfig:
    enabled: bool
    cache_dir: str | None
    version: str
    mode: str
    namespace: str
    warmup: bool


def resolve_materialized_sample_cache_config(
    *,
    puuid_index: Mapping[str, int],
    exclude_match_ids: Sequence[str] | set[str] | None,
    cache_dir: str | None = None,
    version: str | None = None,
    mode: str | None = None,
    warmup: bool | None = None,
    sample_variant: str = "full",
) -> MaterializedSampleCacheConfig:
    resolved_dir = cache_dir or os.environ.get(SAMPLE_CACHE_DIR_ENV)
    resolved_mode = _normalize_mode(mode, resolved_dir)
    if resolved_mode != "off" and not resolved_dir:
        resolved_dir = DEFAULT_SAMPLE_CACHE_DIR
    resolved_version = str(version or os.environ.get(SAMPLE_CACHE_VERSION_ENV) or DEFAULT_SAMPLE_CACHE_VERSION)
    resolved_warmup = _flag_from_env(SAMPLE_CACHE_WARMUP_ENV, default=False) if warmup is None else bool(warmup)
    namespace = _stable_hash_bytes(
        [
            resolved_version.encode("utf-8"),
            sample_variant.encode("utf-8"),
            _puuid_index_signature(puuid_index).encode("utf-8"),
            _exclude_signature(exclude_match_ids).encode("utf-8"),
        ]
    )
    return MaterializedSampleCacheConfig(
        enabled=(resolved_mode != "off"),
        cache_dir=resolved_dir,
        version=resolved_version,
        mode=resolved_mode,
        namespace=namespace,
        warmup=resolved_warmup,
    )


class PlanBSampleCache:
    """Versioned cache for materialized per-match training samples."""

    def __init__(
        self,
        *,
        puuid_index: Mapping[str, int],
        exclude_match_ids: Sequence[str] | set[str] | None,
        cache_dir: str | None = None,
        version: str | None = None,
        mode: str | None = None,
        warmup: bool | None = None,
        sample_variant: str = "full",
    ) -> None:
        self.config = resolve_materialized_sample_cache_config(
            puuid_index=puuid_index,
            exclude_match_ids=exclude_match_ids,
            cache_dir=cache_dir,
            version=version,
            mode=mode,
            warmup=warmup,
            sample_variant=sample_variant,
        )
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def warmup(self) -> bool:
        return self.config.enabled and self.config.warmup

    def cache_path_for(self, match_id: str) -> str:
        if not self.config.cache_dir:
            raise RuntimeError("materialized sample cache directory is not configured")
        return os.path.join(
            self.config.cache_dir,
            self.config.version,
            self.config.namespace,
            f"{match_id}.pt",
        )

    def load(self, match_id: str) -> dict[str, Any] | None:
        if not self.enabled or self.config.mode not in _READ_MODES:
            self.misses += 1
            return None
        path = self.cache_path_for(match_id)
        if not os.path.exists(path):
            self.misses += 1
            return None
        sample = torch.load(path, map_location="cpu", weights_only=False)
        self.hits += 1
        return sample

    def store(self, match_id: str, sample: Mapping[str, Any]) -> None:
        if not self.enabled or self.config.mode not in _WRITE_MODES:
            return
        path = self.cache_path_for(match_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(path),
            prefix=f"{match_id}.",
            suffix=".tmp",
        )
        os.close(fd)
        try:
            torch.save(dict(sample), tmp_path)
            os.replace(tmp_path, path)
            self.writes += 1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def get_or_build(
        self,
        match_id: str,
        builder: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        if self.enabled and self.config.mode != "refresh":
            cached = self.load(match_id)
            if cached is not None:
                return cached
        else:
            self.misses += 1
        sample = builder(match_id)
        self.store(match_id, sample)
        return sample

    def materialize_many(
        self,
        match_ids: Sequence[str],
        builder: Callable[[str], dict[str, Any]],
    ) -> None:
        if not self.enabled or self.config.mode == "readonly":
            return
        for match_id in match_ids:
            self.get_or_build(match_id, builder)

    def stats(self) -> dict[str, int | str]:
        return {
            "mode": self.config.mode,
            "version": self.config.version,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    def namespace_dir(self) -> str | None:
        if not self.config.cache_dir:
            return None
        return os.path.join(self.config.cache_dir, self.config.version, self.config.namespace)

    def materialized_count(self) -> int:
        """Count .pt files on disk for this cache's (version, namespace).

        Used to observe hit state across DataLoader worker processes, whose
        in-memory hit/miss counters are not visible to the parent process.
        """
        ns_dir = self.namespace_dir()
        if not ns_dir or not os.path.isdir(ns_dir):
            return 0
        count = 0
        with os.scandir(ns_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".pt"):
                    count += 1
        return count
