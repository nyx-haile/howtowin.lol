"""Tests for Step 3 / Gate C skill-aware retrieval."""
import torch

from model.retrieval import (
    IndexBundle,
    KEY_DIM,
    SCHEMA_VERSION,
    SKILL_MIN_EFFECTIVE_K,
    Whitener,
    query_index_skill_aware,
    save_index,
    load_index,
)


def _mk_bundle(*, N: int, band_values: list[int], seed: int = 0) -> IndexBundle:
    torch.manual_seed(seed)
    corpus_raw = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus_raw)
    assert len(band_values) == N
    return IndexBundle(
        corpus_white=w.apply(corpus_raw),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.zeros(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
        row_rank_band=torch.tensor(band_values, dtype=torch.int8),
        schema_version=SCHEMA_VERSION,
    ), corpus_raw


def test_skill_aware_query_raises_without_rank_metadata():
    torch.manual_seed(0)
    N = 16
    corpus = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus), whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.zeros(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
    )
    q = torch.randn(2, KEY_DIM)
    try:
        query_index_skill_aware(
            bundle, q, k=4, query_rank_bands=torch.tensor([0, 1]),
        )
    except ValueError as e:
        assert "row_rank_band" in str(e)
    else:
        raise AssertionError("expected ValueError for missing rank metadata")


def test_skill_aware_same_band_when_enough_rows():
    # 200 rows, half in band 0 / half in band 3. Query at band 0 should pull
    # only band-0 rows at k=32 because same-band eligibility >= min_effective_k.
    bands = [0] * 100 + [3] * 100
    bundle, corpus_raw = _mk_bundle(N=200, band_values=bands)
    # Pick a query that is itself one of the band-0 rows, ensuring retrieval
    # does match band-0 rows first.
    q_raw = corpus_raw[0:1]
    idx, d, info = query_index_skill_aware(
        bundle, q_raw, k=32,
        query_rank_bands=torch.tensor([0], dtype=torch.long),
        min_effective_k=SKILL_MIN_EFFECTIVE_K,
    )
    assert idx.shape == (1, 32)
    assert int(info["widening_stage_per_query"][0].item()) == 0   # same band
    assert int(info["effective_k_per_query"][0].item()) == 100
    # 100% of returned rows should be in band 0.
    assert float(info["in_band_fraction_per_query"][0].item()) == 1.0
    assert (bundle.row_rank_band[idx[0]] == 0).all().item()


def test_skill_aware_widens_to_adjacent_when_same_band_too_small():
    # 10 band-0 rows (below min_effective_k=32); 100 band-1, 100 band-3.
    # Band 1 is adjacent to band 0 but band 3 is not. Widening to |diff|<=1
    # should include band-0 + band-1 = 110 rows; returned rows should all be
    # band 0 or 1, not band 3.
    bands = [0] * 10 + [1] * 100 + [3] * 100
    bundle, corpus_raw = _mk_bundle(N=210, band_values=bands)
    q_raw = corpus_raw[0:1]
    idx, d, info = query_index_skill_aware(
        bundle, q_raw, k=32,
        query_rank_bands=torch.tensor([0], dtype=torch.long),
        min_effective_k=32,
    )
    assert int(info["widening_stage_per_query"][0].item()) == 1   # adjacent
    # No band-3 rows should appear.
    chosen_bands = bundle.row_rank_band[idx[0]].tolist()
    assert 3 not in chosen_bands


def test_skill_aware_falls_back_to_unrestricted_with_penalty():
    # 5 band-0, 5 band-1 (both too small for k=32 even adjacent). Unrestricted
    # path should engage; with penalty, band-0 rows should still be preferred
    # for a band-0 query.
    bands = [0] * 5 + [1] * 5 + [3] * 100
    bundle, corpus_raw = _mk_bundle(N=110, band_values=bands)
    q_raw = corpus_raw[0:1]
    idx, d, info = query_index_skill_aware(
        bundle, q_raw, k=32,
        query_rank_bands=torch.tensor([0], dtype=torch.long),
        min_effective_k=32,
        out_of_band_penalty=2.0,   # strong preference signal
    )
    assert int(info["widening_stage_per_query"][0].item()) == 2   # unrestricted
    chosen_bands = bundle.row_rank_band[idx[0]].tolist()
    # With penalty 2.0 and very few close-band rows, the top of the list
    # should contain all 5 band-0 rows (self-query + 4 neighbours).
    assert chosen_bands.count(0) >= 5


def test_skill_aware_unranked_query_matches_all_rows_without_penalty():
    bands = [0] * 50 + [3] * 50
    bundle, corpus_raw = _mk_bundle(N=100, band_values=bands)
    q_raw = corpus_raw[0:1]
    idx, d, info = query_index_skill_aware(
        bundle, q_raw, k=16,
        query_rank_bands=torch.tensor([-1], dtype=torch.long),
        min_effective_k=32,
    )
    # Unranked query stays in same-band stage (no widening needed, all rows
    # match via wildcard).
    assert int(info["widening_stage_per_query"][0].item()) == 0
    assert idx[0, 0].item() == 0  # self-match


def test_skill_aware_handles_empty_query():
    bands = [0] * 10
    bundle, _ = _mk_bundle(N=10, band_values=bands)
    idx, d, info = query_index_skill_aware(
        bundle, torch.zeros(0, KEY_DIM), k=4,
        query_rank_bands=torch.zeros(0, dtype=torch.long),
    )
    assert idx.shape == (0, 4)
    assert d.shape == (0, 4)
    assert info["effective_k_per_query"].shape == (0,)


def test_index_bundle_saves_and_loads_rank_metadata(tmp_path):
    bands = [0, 1, 2, 3, -1, 0, 0, 1]
    bundle, _ = _mk_bundle(N=8, band_values=bands)
    p = tmp_path / "idx.pt"
    save_index(bundle, str(p))
    loaded = load_index(str(p))
    assert loaded.row_rank_band is not None
    assert torch.equal(loaded.row_rank_band, bundle.row_rank_band)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.has_rank_metadata()


def test_legacy_bundle_load_has_no_rank_metadata(tmp_path):
    """An index saved by older code (no schema_version) loads with row_rank_band=None."""
    import torch as _t
    torch.manual_seed(0)
    N = 4
    corpus = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus)
    # Simulate a legacy payload (missing schema_version and row_rank_band).
    payload = {
        "corpus_white": w.apply(corpus),
        "whitener": w.state_dict(),
        "row_match_id": [f"M{i}" for i in range(N)],
        "row_anchor_minute": _t.zeros(N, dtype=_t.int64),
        "row_blue_win": _t.zeros(N, dtype=_t.int8),
        "checkpoint_sha": "x",
        "code_sha": "y",
        "built_at": 0,
    }
    p = tmp_path / "legacy.pt"
    _t.save(payload, str(p))
    loaded = load_index(str(p))
    assert loaded.row_rank_band is None
    assert loaded.schema_version == 1
    assert not loaded.has_rank_metadata()
