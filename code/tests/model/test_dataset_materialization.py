from __future__ import annotations

import os

import numpy as np
import torch

from model.dataset import MatchDataset, build_puuid_index


def _assert_samples_equal(lhs, rhs):
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        left = lhs[key]
        right = rhs[key]
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        elif isinstance(left, np.ndarray):
            assert isinstance(right, np.ndarray)
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, list):
            assert isinstance(right, list)
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    assert left_item == right_item
                else:
                    _assert_samples_equal({"value": left_item}, {"value": right_item})
        else:
            assert left == right


def test_materialized_sample_matches_on_demand_path(fixture_match_id, tmp_path):
    idx = build_puuid_index([fixture_match_id])
    ds_on_demand = MatchDataset([fixture_match_id], puuid_index=idx, cache_size=0)
    ds_materialized = MatchDataset(
        [fixture_match_id],
        puuid_index=idx,
        cache_size=0,
        materialized_cache_dir=str(tmp_path),
        materialized_cache_version="parity",
        materialized_cache_mode="refresh",
    )

    on_demand = ds_on_demand[0]
    materialized = ds_materialized[0]

    _assert_samples_equal(on_demand, materialized)
    cache_path = ds_materialized.materialized_sample_cache.cache_path_for(fixture_match_id)
    assert os.path.exists(cache_path)
    assert cache_path.startswith(str(tmp_path))
    assert os.path.join("parity", ds_materialized.materialized_sample_cache.config.namespace) in cache_path


def test_materialized_sample_cache_hit_skips_rebuild(fixture_match_id, tmp_path):
    idx = build_puuid_index([fixture_match_id])
    ds_write = MatchDataset(
        [fixture_match_id],
        puuid_index=idx,
        cache_size=0,
        materialized_cache_dir=str(tmp_path),
        materialized_cache_version="hits",
        materialized_cache_mode="readwrite",
    )
    first = ds_write[0]
    assert ds_write.materialized_sample_cache.stats()["writes"] == 1

    ds_read = MatchDataset(
        [fixture_match_id],
        puuid_index=idx,
        cache_size=0,
        materialized_cache_dir=str(tmp_path),
        materialized_cache_version="hits",
        materialized_cache_mode="readonly",
    )

    def _unexpected_build(_match_id):
        raise AssertionError("cache miss unexpectedly rebuilt the sample")

    ds_read._build_sample = _unexpected_build
    second = ds_read[0]

    _assert_samples_equal(first, second)
    stats = ds_read.materialized_sample_cache.stats()
    assert stats["hits"] == 1
    assert stats["writes"] == 0


def test_materialized_sample_version_changes_cache_location(fixture_match_id, tmp_path):
    idx = build_puuid_index([fixture_match_id])
    ds_v1 = MatchDataset(
        [fixture_match_id],
        puuid_index=idx,
        materialized_cache_dir=str(tmp_path),
        materialized_cache_version="v1-test",
        materialized_cache_mode="readwrite",
    )
    ds_v2 = MatchDataset(
        [fixture_match_id],
        puuid_index=idx,
        materialized_cache_dir=str(tmp_path),
        materialized_cache_version="v2-test",
        materialized_cache_mode="readwrite",
    )

    assert ds_v1.materialized_sample_cache.cache_path_for(fixture_match_id) != ds_v2.materialized_sample_cache.cache_path_for(fixture_match_id)


def test_core_materialized_sample_omits_legacy_labels_but_still_collates(fixture_match_id, tmp_path):
    idx = build_puuid_index([fixture_match_id])
    ds_core = MatchDataset(
        [fixture_match_id],
        puuid_index=idx,
        cache_size=0,
        materialized_cache_dir=str(tmp_path),
        materialized_cache_version="core",
        materialized_cache_mode="readwrite",
        include_legacy_labels=False,
    )

    sample = ds_core[0]
    assert "labels" not in sample
    assert "label_mask" not in sample

    cache_path = ds_core.materialized_sample_cache.cache_path_for(fixture_match_id)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert "labels" not in cached
    assert "label_mask" not in cached

    from model.dataset import collate_games

    batch = collate_games([sample])
    assert batch["labels"].shape[:2] == batch["tokens"].shape[:2]
    assert batch["label_mask"].shape == batch["tokens"].shape
    assert torch.count_nonzero(batch["labels"]) == 0
    assert torch.count_nonzero(batch["label_mask"]) == 0
