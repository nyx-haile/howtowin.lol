"""Tests for model.eval_cache: round-trip, cache invalidation, atomic writes."""
import json
import os
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(tmp_path, monkeypatch):
    """Patch CACHE_DIR to tmp_path and return the module."""
    import model.eval_cache as ec
    monkeypatch.setattr(ec, "_CACHE_DIR", Path(tmp_path))
    return ec


def _fake_keys(n: int = 10, dim: int = 544):
    keys = torch.randn(n, dim)
    minutes = torch.randint(10, 26, (n,), dtype=torch.int64)
    return keys, minutes


# ---------------------------------------------------------------------------
# Round-trip: save → load
# ---------------------------------------------------------------------------

def test_save_load_keys_roundtrip(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys()
    ids = ["match_a", "match_b", "match_c"]

    ec.save_cached_keys("model", "game_cold",
                        checkpoint_sha="abc123",
                        holdout_match_ids=ids,
                        keys=keys, minutes=minutes)

    result = ec.load_cached_keys("model", "game_cold",
                                 checkpoint_sha="abc123",
                                 holdout_match_ids=ids)
    assert result is not None
    assert torch.equal(result["keys"], keys)
    assert torch.equal(result["minutes"], minutes)


def test_save_writes_manifest_sidecar(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys(n=5)
    ids = ["m1"]

    ec.save_cached_keys("model", "game_cold",
                        checkpoint_sha="deadbeef",
                        holdout_match_ids=ids,
                        keys=keys, minutes=minutes)

    manifests = list(Path(tmp_path).glob("*.manifest.json"))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())
    assert m["n_keys"] == 5
    assert m["checkpoint_sha"] == "deadbeef"
    assert m["kind"] == "model"
    assert "source_hash" in m
    assert "encoded_at" in m


def test_save_load_cohort_idx_roundtrip(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    cohort_idx = torch.randint(0, 1000, (20, 64))
    ids = ["x", "y"]

    ec.save_cached_cohort_idx("model", "player_cold",
                              checkpoint_sha="sha1",
                              holdout_match_ids=ids,
                              cohort_idx=cohort_idx)

    loaded = ec.load_cached_cohort_idx("model", "player_cold",
                                       checkpoint_sha="sha1",
                                       holdout_match_ids=ids)
    assert loaded is not None
    assert torch.equal(loaded, cohort_idx)


# ---------------------------------------------------------------------------
# Cache miss on unknown key
# ---------------------------------------------------------------------------

def test_load_returns_none_on_miss(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    result = ec.load_cached_keys("model", "game_cold",
                                 checkpoint_sha="notcached",
                                 holdout_match_ids=["m1"])
    assert result is None


def test_load_cohort_idx_returns_none_on_miss(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    result = ec.load_cached_cohort_idx("model", "game_cold",
                                       checkpoint_sha="notcached",
                                       holdout_match_ids=["m1"])
    assert result is None


# ---------------------------------------------------------------------------
# Cache invalidation: one axis at a time
# ---------------------------------------------------------------------------

def test_cache_misses_on_checkpoint_sha_change(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys()
    ids = ["match_a"]

    ec.save_cached_keys("model", "game_cold",
                        checkpoint_sha="sha_v1",
                        holdout_match_ids=ids,
                        keys=keys, minutes=minutes)

    # Same ids, different checkpoint → miss.
    result = ec.load_cached_keys("model", "game_cold",
                                 checkpoint_sha="sha_v2",
                                 holdout_match_ids=ids)
    assert result is None


def test_cache_misses_on_holdout_list_change(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys()
    ids_orig = ["match_a", "match_b"]

    ec.save_cached_keys("model", "game_cold",
                        checkpoint_sha="sha_v1",
                        holdout_match_ids=ids_orig,
                        keys=keys, minutes=minutes)

    # One extra match added → miss.
    result = ec.load_cached_keys("model", "game_cold",
                                 checkpoint_sha="sha_v1",
                                 holdout_match_ids=ids_orig + ["match_c"])
    assert result is None


def test_cache_misses_on_source_bytes_change(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys()
    ids = ["match_a"]

    # Save with the real source hash.
    ec.save_cached_keys("model", "game_cold",
                        checkpoint_sha="sha_v1",
                        holdout_match_ids=ids,
                        keys=keys, minutes=minutes)

    # Patch compute_source_hash to return a different digest, simulating an
    # encoder source change.
    monkeypatch.setattr(ec, "compute_source_hash", lambda kind: "totally_different_hash")
    result = ec.load_cached_keys("model", "game_cold",
                                 checkpoint_sha="sha_v1",
                                 holdout_match_ids=ids)
    assert result is None


# ---------------------------------------------------------------------------
# frame_features cache is checkpoint-independent
# ---------------------------------------------------------------------------

def test_frame_features_cache_is_checkpoint_independent(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys(dim=91)
    ids = ["gm1", "gm2"]

    ec.save_cached_keys("frame_features", "game_cold",
                        checkpoint_sha=None,
                        holdout_match_ids=ids,
                        keys=keys, minutes=minutes)

    # Any (or no) checkpoint_sha should hit the same cache.
    for sha in (None, "sha_v1", "sha_v2"):
        result = ec.load_cached_keys("frame_features", "game_cold",
                                     checkpoint_sha=sha,
                                     holdout_match_ids=ids)
        assert result is not None, f"expected HIT for checkpoint_sha={sha!r}"
        assert torch.equal(result["keys"], keys)


# ---------------------------------------------------------------------------
# delete_cached removes all three files
# ---------------------------------------------------------------------------

def test_delete_cached_removes_all_files(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys()
    cohort_idx = torch.randint(0, 100, (10, 16))
    ids = ["del_me"]

    ec.save_cached_keys("model", "game_cold",
                        checkpoint_sha="sha",
                        holdout_match_ids=ids,
                        keys=keys, minutes=minutes)
    ec.save_cached_cohort_idx("model", "game_cold",
                              checkpoint_sha="sha",
                              holdout_match_ids=ids,
                              cohort_idx=cohort_idx)

    before = list(Path(tmp_path).iterdir())
    assert len(before) >= 2  # .pt + .manifest.json + cohort_idx.pt

    n = ec.delete_cached("model", "game_cold",
                         checkpoint_sha="sha",
                         holdout_match_ids=ids)
    assert n >= 2

    after = list(Path(tmp_path).iterdir())
    assert len(after) == 0


# ---------------------------------------------------------------------------
# Atomic write: partial write doesn't leave corrupt files
# ---------------------------------------------------------------------------

def test_atomic_save_leaves_no_tmp_on_success(tmp_path, monkeypatch):
    ec = _make_cache(tmp_path, monkeypatch)
    keys, minutes = _fake_keys()
    ids = ["atomictest"]

    ec.save_cached_keys("model", "game_cold",
                        checkpoint_sha="sha",
                        holdout_match_ids=ids,
                        keys=keys, minutes=minutes)

    tmps = list(Path(tmp_path).glob("*.tmp"))
    assert len(tmps) == 0
