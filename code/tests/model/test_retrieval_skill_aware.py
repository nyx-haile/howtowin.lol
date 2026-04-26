"""Tests for Step 3 / Gate C skill-aware retrieval."""
import torch

import model.retrieval as retrieval
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


def _legacy_query_index_skill_aware(
    bundle: IndexBundle,
    queries_raw: torch.Tensor,
    *,
    k: int,
    query_rank_bands: torch.Tensor,
    min_effective_k: int,
    out_of_band_penalty: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Reference copy of the pre-short-circuit per-k semantics.

    The optimization under test may skip fallback stage ``topk`` calls only
    when the chosen same-stage result is provably unchanged.
    """
    corpus = bundle.corpus_white
    row_band = bundle.row_rank_band.to(torch.long)
    valid_row = row_band >= 0
    inv_valid = (~valid_row).unsqueeze(0)
    not_excluded = torch.ones(1, corpus.shape[0], dtype=torch.bool)

    qb_white = bundle.whitener.apply(queries_raw.to(dtype=corpus.dtype))
    d_full = torch.cdist(qb_white, corpus)
    q_band = query_rank_bands.to(dtype=torch.long)
    unranked_q = (q_band < 0).unsqueeze(1)
    diff = (row_band.unsqueeze(0) - q_band.unsqueeze(1)).abs()

    same_mask = unranked_q | inv_valid | (diff == 0)
    adj_mask = unranked_q | inv_valid | (diff <= 1)
    same_elig = same_mask & not_excluded
    adj_elig = adj_mask & not_excluded

    penalty_t = torch.tensor(out_of_band_penalty, dtype=torch.float32)
    penalty_pow = torch.pow(penalty_t, diff.to(d_full.dtype))
    ones = torch.ones_like(penalty_pow)
    dist_penalty = torch.where(
        valid_row.unsqueeze(0).expand_as(penalty_pow), penalty_pow, ones,
    )
    dist_penalty = torch.where(unranked_q.expand_as(penalty_pow), ones, dist_penalty)

    d_same = torch.where(same_elig, d_full, torch.full_like(d_full, float("inf")))
    d_adj = torch.where(adj_elig, d_full, torch.full_like(d_full, float("inf")))
    d_unr = torch.where(not_excluded.expand_as(d_full), d_full * dist_penalty, torch.full_like(d_full, float("inf")))

    d_same_top, idx_same_top = d_same.topk(k, dim=1, largest=False)
    d_adj_top, idx_adj_top = d_adj.topk(k, dim=1, largest=False)
    d_unr_top, idx_unr_top = d_unr.topk(k, dim=1, largest=False)

    eff_k = same_mask.sum(dim=1).to(torch.long)
    same_kept = torch.clamp(same_elig.sum(dim=1), max=k)
    adj_kept = torch.clamp(adj_elig.sum(dim=1), max=k)
    ranked_q = ~unranked_q.squeeze(1)
    escalate_to_adj = ranked_q & (same_kept < min_effective_k)
    escalate_to_unr = escalate_to_adj & (adj_kept < min_effective_k)
    stage = torch.where(
        escalate_to_unr,
        torch.full_like(escalate_to_unr, 2, dtype=torch.long),
        torch.where(
            escalate_to_adj,
            torch.full_like(escalate_to_adj, 1, dtype=torch.long),
            torch.full_like(ranked_q, 0, dtype=torch.long),
        ),
    )

    stage_exp = stage.unsqueeze(1).expand(-1, k)
    chosen_idx = torch.where(
        stage_exp == 2,
        idx_unr_top,
        torch.where(stage_exp == 1, idx_adj_top, idx_same_top),
    )
    chosen_d = torch.where(
        stage_exp == 2,
        d_unr_top,
        torch.where(stage_exp == 1, d_adj_top, d_same_top),
    )

    if torch.isinf(chosen_d).any():
        d_full_top, idx_full_top = d_full.topk(min(k, corpus.shape[0]), dim=1, largest=False)
        if d_full_top.shape[1] < k:
            pad = k - d_full_top.shape[1]
            d_full_top = torch.cat([d_full_top, d_full_top[:, -1:].expand(-1, pad)], dim=1)
            idx_full_top = torch.cat([idx_full_top, idx_full_top[:, -1:].expand(-1, pad)], dim=1)
        inf_row = torch.isinf(chosen_d).any(dim=1)
        chosen_idx = torch.where(inf_row.unsqueeze(1), idx_full_top, chosen_idx)
        chosen_d = torch.where(inf_row.unsqueeze(1), d_full_top, chosen_d)
        stage = torch.where(inf_row, torch.full_like(stage, 2), stage)

    chosen_band = row_band[chosen_idx]
    in_band = (chosen_band == q_band.unsqueeze(1)).to(torch.float32).mean(dim=1)
    in_band = torch.where(
        unranked_q.squeeze(1),
        torch.full_like(in_band, float("nan")),
        in_band,
    )
    return chosen_idx, chosen_d, {
        "effective_k_per_query": eff_k,
        "widening_stage_per_query": stage,
        "in_band_fraction_per_query": in_band,
    }


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


def test_skill_aware_same_only_batch_skips_fallback_topk(monkeypatch):
    bands = [0] * 80 + [3] * 80
    bundle, corpus_raw = _mk_bundle(N=160, band_values=bands)

    calls: list[int] = []
    original_topk = retrieval._topk_smallest

    def counted_topk(d: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(k)
        return original_topk(d, k)

    monkeypatch.setattr(retrieval, "_topk_smallest", counted_topk)
    idx, _d, info = retrieval.query_index_skill_aware(
        bundle, corpus_raw[:2], k=32,
        query_rank_bands=torch.tensor([0, 0], dtype=torch.long),
        min_effective_k=32,
        batch_size=8,
    )

    assert idx.shape == (2, 32)
    assert info["widening_stage_per_query"].tolist() == [0, 0]
    assert len(calls) == 1


def test_skill_aware_matches_legacy_semantics_on_mixed_batch():
    bands = [0] * 40 + [2] * 10 + [4] * 80 + [-1] * 2
    bundle, corpus_raw = _mk_bundle(N=132, band_values=bands)
    q_raw = torch.cat([
        corpus_raw[0:1],    # band-0 query: same stage
        corpus_raw[50:51],  # synthetic band-1 query: adjacent stage
        corpus_raw[40:41],  # band-2 query: unrestricted stage
        corpus_raw[0:1],    # unranked query: wildcard same stage
    ], dim=0)
    q_band = torch.tensor([0, 1, 2, -1], dtype=torch.long)

    actual_idx, actual_d, actual_info = query_index_skill_aware(
        bundle, q_raw, k=32, query_rank_bands=q_band,
        min_effective_k=32, batch_size=4,
    )
    expected_idx, expected_d, expected_info = _legacy_query_index_skill_aware(
        bundle, q_raw, k=32, query_rank_bands=q_band,
        min_effective_k=32,
    )

    assert torch.equal(actual_idx, expected_idx)
    assert torch.allclose(actual_d, expected_d)
    assert torch.equal(
        actual_info["effective_k_per_query"],
        expected_info["effective_k_per_query"],
    )
    assert actual_info["widening_stage_per_query"].tolist() == [0, 1, 2, 0]
    assert torch.equal(
        actual_info["widening_stage_per_query"],
        expected_info["widening_stage_per_query"],
    )
    assert torch.allclose(
        actual_info["in_band_fraction_per_query"],
        expected_info["in_band_fraction_per_query"],
        equal_nan=True,
    )


def test_skill_aware_k_below_min_effective_preserves_unrestricted_stage():
    bands = [0] * 80 + [3] * 80
    bundle, corpus_raw = _mk_bundle(N=160, band_values=bands)
    idx, d, info = query_index_skill_aware(
        bundle, corpus_raw[0:1], k=16,
        query_rank_bands=torch.tensor([0], dtype=torch.long),
        min_effective_k=32,
    )
    expected_idx, expected_d, expected_info = _legacy_query_index_skill_aware(
        bundle, corpus_raw[0:1], k=16,
        query_rank_bands=torch.tensor([0], dtype=torch.long),
        min_effective_k=32,
    )

    assert int(info["widening_stage_per_query"][0].item()) == 2
    assert torch.equal(idx, expected_idx)
    assert torch.allclose(d, expected_d)
    assert torch.equal(
        info["effective_k_per_query"],
        expected_info["effective_k_per_query"],
    )


def test_skill_aware_unranked_corpus_rows_are_same_stage_wildcards():
    bands = [0] * 30 + [-1] * 2 + [3] * 100
    bundle, corpus_raw = _mk_bundle(N=132, band_values=bands)
    idx, _d, info = query_index_skill_aware(
        bundle, corpus_raw[0:1], k=32,
        query_rank_bands=torch.tensor([0], dtype=torch.long),
        min_effective_k=32,
    )

    chosen_bands = bundle.row_rank_band[idx[0]]
    assert int(info["widening_stage_per_query"][0].item()) == 0
    assert int(info["effective_k_per_query"][0].item()) == 32
    assert int((chosen_bands == -1).sum().item()) == 2


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
