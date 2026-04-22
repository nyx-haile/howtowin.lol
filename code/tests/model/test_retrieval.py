import torch
import pytest


def test_whitener_fit_makes_corpus_unit_normal():
    from model.retrieval import Whitener
    torch.manual_seed(0)
    corpus = torch.randn(1024, 32) * 7.5 + 3.0
    w = Whitener.fit(corpus)
    out = w.apply(corpus)
    assert torch.allclose(out.mean(dim=0), torch.zeros(32), atol=1e-5)
    assert torch.allclose(out.std(dim=0, unbiased=False),
                          torch.ones(32), atol=1e-3)


def test_whitener_apply_uses_fitted_params_not_query_stats():
    from model.retrieval import Whitener
    # Deterministic corpus: mean=0, std=1, per dim.
    corpus = torch.tensor(
        [[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]]
    )
    w = Whitener.fit(corpus)
    # Query with values whose own mean is 100 — distinct from corpus mean 0.
    # If apply() wrongly used query-stats it would map these to zero.
    # With fitted params: (100 - 0) / 1 = 100.
    query = torch.full((3, 2), 100.0)
    out = w.apply(query)
    assert torch.allclose(out, torch.full((3, 2), 100.0), atol=1e-5)


def test_whitener_clamps_zero_std_dims():
    from model.retrieval import Whitener
    corpus = torch.zeros(100, 4)
    corpus[:, 0] = torch.arange(100, dtype=torch.float32)  # nonzero std
    # Dims 1, 2, 3 are constant → std==0; whitener must not produce NaN/Inf.
    w = Whitener.fit(corpus)
    out = w.apply(corpus)
    assert torch.isfinite(out).all()


def test_whitener_roundtrips_through_state_dict():
    from model.retrieval import Whitener
    torch.manual_seed(0)
    corpus = torch.randn(64, 16)
    w = Whitener.fit(corpus)
    sd = w.state_dict()
    w2 = Whitener.from_state_dict(sd)
    assert torch.equal(w.mu, w2.mu)
    assert torch.equal(w.sigma, w2.sigma)


def test_encode_game_keys_shapes(fixture_match_id):
    from model.retrieval import encode_game_keys, KEY_DIM
    from model.plan_b_model import PlanBModel
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    model = PlanBModel(max_puuids=len(idx) + 1)
    keys, minutes, blue_win = encode_game_keys(model, batch)
    T = batch["anchor_positions"].size(1)
    assert keys.shape == (T, KEY_DIM)
    assert keys.dtype == torch.float32
    assert minutes.shape == (T,)
    assert minutes.dtype == torch.int64
    assert blue_win.shape == ()         # scalar tensor
    assert blue_win.dtype == torch.int8
    assert int(blue_win.item()) in (0, 1)


def test_encode_game_keys_minutes_are_monotonic(fixture_match_id):
    from model.retrieval import encode_game_keys
    from model.plan_b_model import PlanBModel
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    model = PlanBModel(max_puuids=len(idx) + 1)
    _, minutes, _ = encode_game_keys(model, batch)
    diffs = minutes[1:] - minutes[:-1]
    assert (diffs >= 0).all(), \
        "anchor minutes must be non-decreasing"


def test_index_bundle_roundtrips(tmp_path):
    from model.retrieval import IndexBundle, Whitener, KEY_DIM, save_index, load_index
    torch.manual_seed(0)
    N = 32
    corpus = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.arange(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="deadbeef",
        code_sha="cafef00d",
        built_at=1700000000,
    )
    p = tmp_path / "idx.pt"
    save_index(bundle, str(p))
    loaded = load_index(str(p))
    assert torch.equal(bundle.corpus_white, loaded.corpus_white)
    assert torch.equal(bundle.whitener.mu, loaded.whitener.mu)
    assert torch.equal(bundle.whitener.sigma, loaded.whitener.sigma)
    assert bundle.row_match_id == loaded.row_match_id
    assert torch.equal(bundle.row_anchor_minute, loaded.row_anchor_minute)
    assert torch.equal(bundle.row_blue_win, loaded.row_blue_win)
    assert loaded.checkpoint_sha == "deadbeef"
    assert loaded.code_sha == "cafef00d"
    assert loaded.built_at == 1700000000


def test_build_index_excludes_holdout_match_ids(fixture_match_id):
    from model.retrieval import build_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids={fixture_match_id},
        puuid_index=idx,
        device="cpu",
    )
    # The only candidate match was excluded — corpus should be empty.
    assert bundle.corpus_white.shape[0] == 0
    assert bundle.row_match_id == []


def test_build_index_corpus_is_whitened(fixture_match_id):
    from model.retrieval import build_index, KEY_DIM
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    torch.manual_seed(0)
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
    )
    N = bundle.corpus_white.shape[0]
    assert N > 0
    assert bundle.corpus_white.shape == (N, KEY_DIM)
    # With one game, whitening forces every row identical → mean ~ 0.
    assert torch.allclose(
        bundle.corpus_white.mean(dim=0), torch.zeros(KEY_DIM), atol=1e-4
    )


def test_build_index_whitening_is_stable_for_known_random_seeds(fixture_match_id):
    from model.retrieval import build_index, KEY_DIM
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])

    for seed in (0, 40, 44):
        torch.manual_seed(seed)
        model = PlanBModel(max_puuids=len(idx) + 1)
        bundle = build_index(
            model=model,
            train_match_ids=[fixture_match_id],
            exclude_match_ids=set(),
            puuid_index=idx,
            device="cpu",
        )
        assert bundle.corpus_white.shape[1] == KEY_DIM
        assert (
            bundle.corpus_white.mean(dim=0).abs().max().item() < 1e-4
        ), f"whitening drifted above tolerance for seed {seed}"


def test_build_index_records_per_row_metadata(fixture_match_id):
    from model.retrieval import build_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
    )
    N = bundle.corpus_white.shape[0]
    # All rows come from the one fixture match.
    assert all(mid == fixture_match_id for mid in bundle.row_match_id)
    assert len(bundle.row_match_id) == N
    assert bundle.row_anchor_minute.shape == (N,)
    assert bundle.row_blue_win.shape == (N,)
    # All rows share the same per-game outcome.
    assert int(bundle.row_blue_win.unique().numel()) == 1


def test_query_index_returns_self_at_rank_one_for_corpus_rows():
    from model.retrieval import (
        IndexBundle, Whitener, KEY_DIM, query_index,
    )
    torch.manual_seed(0)
    N = 100
    corpus_raw = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus_raw)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus_raw),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.arange(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
    )
    queries_raw = corpus_raw[:5]  # exact match against corpus rows 0..4
    cohort_idx, _dists = query_index(bundle, queries_raw, k=3,
                                     device="cpu", batch_size=16)
    assert cohort_idx.shape == (5, 3)
    # Top-1 for each query must be the matching row.
    assert torch.equal(cohort_idx[:, 0], torch.arange(5, dtype=cohort_idx.dtype))


def test_query_index_returns_distinct_cohorts():
    from model.retrieval import (
        IndexBundle, Whitener, KEY_DIM, query_index,
    )
    torch.manual_seed(1)
    N = 64
    corpus_raw = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus_raw)
    bundle = IndexBundle(
        corpus_white=w.apply(corpus_raw),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.zeros(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
    )
    q = torch.randn(8, KEY_DIM)
    cohort_idx, dists = query_index(bundle, q, k=4, device="cpu", batch_size=4)
    assert cohort_idx.shape == (8, 4)
    assert dists.shape == (8, 4)
    # Distances must be monotonically non-decreasing across the k axis.
    diffs = dists[:, 1:] - dists[:, :-1]
    assert (diffs >= -1e-5).all()
