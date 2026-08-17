import math
from types import SimpleNamespace

import torch


def test_binary_entropy_at_half_is_one_bit():
    from model.m4_eval import binary_entropy
    p = torch.tensor([0.5])
    h = binary_entropy(p)
    assert torch.allclose(h, torch.tensor([1.0]), atol=1e-6)


def test_binary_entropy_at_endpoints_is_zero():
    from model.m4_eval import binary_entropy
    p = torch.tensor([0.0, 1.0])
    h = binary_entropy(p)
    assert torch.allclose(h, torch.tensor([0.0, 0.0]), atol=1e-6)


def test_binary_entropy_is_symmetric_around_half():
    from model.m4_eval import binary_entropy
    a = binary_entropy(torch.tensor([0.2]))
    b = binary_entropy(torch.tensor([0.8]))
    assert torch.allclose(a, b, atol=1e-6)


def test_binary_entropy_at_seventy_thirty_matches_formula():
    from model.m4_eval import binary_entropy
    p = 0.7
    expected = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    h = binary_entropy(torch.tensor([p]))
    assert abs(h.item() - expected) < 1e-6


def test_cohort_entropies_pure_cohort_has_zero_entropy():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3]])  # (1 query, k=4)
    blue_win = torch.tensor([1, 1, 1, 1], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert h.shape == (1,)
    assert abs(h[0].item() - 0.0) < 1e-6


def test_cohort_entropies_balanced_cohort_has_one_bit():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3]])
    blue_win = torch.tensor([1, 1, 0, 0], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert abs(h[0].item() - 1.0) < 1e-6


def test_cohort_entropies_per_query_independent():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    blue_win = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert h.shape == (2,)
    assert abs(h[0].item() - 0.0) < 1e-6  # all wins
    assert abs(h[1].item() - 1.0) < 1e-6  # half-half


def test_mean_entropy_at_k_averages():
    from model.m4_eval import mean_entropy_at_k
    cohort_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    blue_win = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int8)
    mean_h = mean_entropy_at_k(cohort_idx, blue_win)
    assert abs(mean_h - 0.5) < 1e-6


def test_random_k_cohort_indices_shape_and_range():
    from model.m4_eval import random_k_cohort_indices
    torch.manual_seed(0)
    idx = random_k_cohort_indices(Q=10, k=4, N=100, seed=0)
    assert idx.shape == (10, 4)
    assert (idx >= 0).all() and (idx < 100).all()


def test_random_k_cohort_indices_deterministic_under_seed():
    from model.m4_eval import random_k_cohort_indices
    a = random_k_cohort_indices(Q=10, k=4, N=100, seed=42)
    b = random_k_cohort_indices(Q=10, k=4, N=100, seed=42)
    assert torch.equal(a, b)


def test_run_m4_eval_random_baseline_uses_seed_offset_per_k(monkeypatch):
    import model.m4_eval as m4

    row_blue_win = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int8)
    queries = torch.zeros(3, 1)
    query_minutes = torch.tensor([10, 11, 12], dtype=torch.int64)
    calls: list[tuple[int, int, int, int]] = []

    monkeypatch.setattr(
        m4,
        "_build_holdout_queries",
        lambda *args, **kwargs: (queries, query_minutes),
    )
    monkeypatch.setattr(
        m4,
        "_eval_one_index",
        lambda *args, **kwargs: {
            "k_sweep": {},
            "per_minute_at_headline_k": {},
            "headline_cohort_entropies": torch.zeros(0),
        },
    )

    def fake_random_k_cohort_indices(*, Q: int, k: int, N: int, seed: int):
        calls.append((Q, k, N, seed))
        return torch.arange(k, dtype=torch.long).repeat(Q, 1)

    monkeypatch.setattr(
        m4, "random_k_cohort_indices", fake_random_k_cohort_indices,
    )

    result = m4.run_m4_eval(
        model=object(),
        model_bundle=SimpleNamespace(row_blue_win=row_blue_win),
        holdout_match_ids=["holdout"],
        holdout_label="seed_check",
        puuid_index={},
        exclude_match_ids=set(),
        k_sweep=(2, 4),
        headline_k=4,
        device="cpu",
        run_baselines=True,
        random_seed=100,
    )

    assert calls == [(3, 2, 6, 102), (3, 4, 6, 104)]
    assert set(result["random"]["k_sweep"]) == {2, 4}


def test_per_minute_entropy_table_groups_by_query_minute():
    from model.m4_eval import per_minute_entropy_table
    cohort_h = torch.tensor([0.5, 0.7, 0.9, 1.0])
    query_minutes = torch.tensor([10, 10, 15, 20])
    table = per_minute_entropy_table(cohort_h, query_minutes,
                                     minutes=(10, 15, 20))
    # float32 mean → Python float: cannot use exact equality.
    assert abs(table[10] - (0.5 + 0.7) / 2) < 1e-6
    assert abs(table[15] - 0.9) < 1e-6
    assert abs(table[20] - 1.0) < 1e-6


def _strict_distance_bundle():
    from model.retrieval import IndexBundle, KEY_DIM, Whitener

    corpus = torch.zeros(6, KEY_DIM)
    corpus[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    return IndexBundle(
        corpus_white=corpus,
        whitener=Whitener(
            mu=torch.zeros(KEY_DIM, dtype=torch.float64),
            sigma=torch.ones(KEY_DIM, dtype=torch.float64),
        ),
        row_match_id=[f"M{i}" for i in range(corpus.shape[0])],
        row_anchor_minute=torch.arange(corpus.shape[0], dtype=torch.int64),
        row_blue_win=torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.int8),
        checkpoint_sha="x",
        code_sha="y",
        built_at=0,
    )


def test_query_index_wide_prefix_matches_repeated_strict_topk():
    from model.retrieval import KEY_DIM, query_index

    bundle = _strict_distance_bundle()
    queries = torch.zeros(2, KEY_DIM)
    queries[:, 0] = torch.tensor([0.2, 3.7])

    wide_idx, wide_d = query_index(
        bundle, queries, k=5, device="cpu", batch_size=1,
    )
    for k in (1, 3, 5):
        idx, dists = query_index(
            bundle, queries, k=k, device="cpu", batch_size=1,
        )
        assert torch.equal(wide_idx[:, :k], idx)
        assert torch.allclose(wide_d[:, :k], dists)


def test_eval_one_index_reuses_one_wide_query_for_k_sweep(monkeypatch):
    import model.m4_eval as m4
    from model.retrieval import KEY_DIM, query_index as real_query_index

    bundle = _strict_distance_bundle()
    queries = torch.zeros(2, KEY_DIM)
    queries[:, 0] = torch.tensor([0.2, 3.7])
    query_minutes = torch.tensor([10, 11], dtype=torch.int64)

    expected: dict[int, tuple[float, torch.Tensor]] = {}
    for k in (2, 4):
        idx, _ = real_query_index(
            bundle, queries, k=k, device="cpu", batch_size=1,
        )
        expected[k] = (
            m4.mean_entropy_at_k(idx, bundle.row_blue_win),
            m4.cohort_entropies(idx, bundle.row_blue_win),
        )

    calls: list[int] = []

    def counting_query_index(bundle, queries_raw, *, k, **kwargs):
        calls.append(int(k))
        return real_query_index(bundle, queries_raw, k=k, **kwargs)

    monkeypatch.setattr(m4, "query_index", counting_query_index)

    result = m4._eval_one_index(
        bundle,
        queries,
        query_minutes,
        k_sweep=(2, 4),
        headline_k=4,
        headline_minutes=(10, 11),
        device="cpu",
        query_batch_size=1,
    )

    assert calls == [4]
    for k in (2, 4):
        assert abs(result["k_sweep"][k] - expected[k][0]) < 1e-6
    assert torch.allclose(
        result["headline_cohort_entropies"], expected[4][1],
    )
    assert abs(
        result["per_minute_at_headline_k"][10] - expected[4][1][0].item()
    ) < 1e-6
    assert abs(
        result["per_minute_at_headline_k"][11] - expected[4][1][1].item()
    ) < 1e-6


def test_run_m4_eval_returns_structured_result(fixture_match_id):
    from model.m4_eval import run_m4_eval
    from model.retrieval import build_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index

    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_index(
        model=model, train_match_ids=[fixture_match_id],
        exclude_match_ids=set(), puuid_index=idx, device="cpu",
    )
    result = run_m4_eval(
        model=model, model_bundle=bundle,
        holdout_match_ids=[fixture_match_id],
        holdout_label="self_eval",
        puuid_index=idx,
        exclude_match_ids=set(),
        k_sweep=(2, 4),
        headline_k=4,
        device="cpu",
        run_baselines=False,
    )
    assert "model" in result
    assert "k_sweep" in result["model"]
    assert "per_minute_at_headline_k" in result["model"]
    for k in (2, 4):
        assert k in result["model"]["k_sweep"]
        v = result["model"]["k_sweep"][k]
        assert 0.0 <= v <= 1.0


def test_cohort_minute_distribution_diagonal_dominant():
    from model.m4_eval import cohort_minute_distribution

    corpus_minute = torch.tensor([10] * 50 + [20] * 50, dtype=torch.int64)
    # All cohort neighbors come from minute-10 rows (indices 0-49)
    cohort_idx = torch.randint(0, 50, (10, 8))
    query_minutes = torch.tensor([10] * 10, dtype=torch.int64)

    dist = cohort_minute_distribution(cohort_idx, corpus_minute, query_minutes, minutes=[10, 20])

    assert dist["same_minute_share_overall"] > 0.9
    hm = dist["heatmap"]
    assert hm[0, 0].item() > 0.9   # minute-10 queries → minute-10 cohort
    assert hm[0, 1].item() < 0.1   # minute-10 queries → minute-20 cohort
    assert abs(hm[0, 0].item() + hm[0, 1].item() - 1.0) < 1e-4


def test_cohort_minute_distribution_broadly_mixed():
    from model.m4_eval import cohort_minute_distribution

    corpus_minute = torch.tensor([10] * 50 + [20] * 50, dtype=torch.int64)
    # Each cohort is exactly half minute-10 and half minute-20
    cohort_idx = torch.cat([
        torch.randint(0, 50, (10, 4)),
        torch.randint(50, 100, (10, 4)),
    ], dim=1)
    query_minutes = torch.tensor([10] * 10, dtype=torch.int64)

    dist = cohort_minute_distribution(cohort_idx, corpus_minute, query_minutes, minutes=[10, 20])

    assert abs(dist["same_minute_share_overall"] - 0.5) < 0.1
    hm = dist["heatmap"]
    assert abs(hm[0, 0].item() - 0.5) < 0.1
    assert abs(hm[0, 1].item() - 0.5) < 0.1


def test_cohort_minute_distribution_mass_conservation():
    from model.m4_eval import cohort_minute_distribution

    corpus_minute = torch.tensor([10] * 30 + [15] * 30 + [20] * 40, dtype=torch.int64)
    cohort_idx = torch.randint(0, 100, (20, 16))
    query_minutes = torch.tensor([10] * 10 + [15] * 5 + [20] * 5, dtype=torch.int64)

    dist = cohort_minute_distribution(cohort_idx, corpus_minute, query_minutes, minutes=[10, 15, 20])
    hm = dist["heatmap"]
    for i, m in enumerate([10, 15, 20]):
        if (query_minutes == m).any():
            assert abs(hm[i].sum().item() - 1.0) < 1e-4


def test_cohort_entropy_by_minute_match_same_vs_cross():
    from model.m4_eval import cohort_entropy_by_minute_match

    # 10 queries at minute 10; corpus: first 32 rows are minute-10 (all blue_win=1),
    # last 32 rows are minute-20 (50/50 blue_win).
    corpus_minute = torch.tensor([10] * 32 + [20] * 32, dtype=torch.int64)
    blue_win = torch.tensor([1] * 32 + [1, 0] * 16, dtype=torch.int8)
    # Cohorts: first 32 neighbors from minute-10 (all win), last 32 from minute-20 (mixed)
    cohort_idx = torch.cat([
        torch.arange(32).unsqueeze(0).expand(10, -1),
        torch.arange(32, 64).unsqueeze(0).expand(10, -1),
    ], dim=1)  # (10, 64)
    query_minutes = torch.tensor([10] * 10, dtype=torch.int64)

    result = cohort_entropy_by_minute_match(
        cohort_idx, corpus_minute, query_minutes, blue_win, minutes=[10, 20], k_min=4,
    )
    # same-minute (minute-10) neighbors are all wins → entropy ≈ 0
    assert result[10]["same_h"] < 0.1
    # cross-minute (minute-20) neighbors are 50/50 → entropy ≈ 1 bit
    assert result[10]["cross_h"] > 0.9
    # minute-20 queries: no queries exist → nan
    assert math.isnan(result[20]["same_h"])


def test_encode_game_keys_eval_is_stochastic_without_rng_replay():
    from model.dataset import FRAME_FEAT_DIM, MACRO_FEAT_DIM
    from model.plan_b_model import PlanBModel
    from model.player_features import PLAYER_FEATURE_DIM
    from model.retrieval import encode_game_keys
    from model.static_features import STATIC_VECTOR_DIM
    from model.tokens import ANCHOR_TOKEN, BOS_TOKEN

    torch.manual_seed(0)
    batch = {
        "static": torch.zeros(1, STATIC_VECTOR_DIM),
        "players": torch.zeros(1, 10, PLAYER_FEATURE_DIM),
        "player_ids": torch.zeros(1, 10, dtype=torch.long),
        "tokens": torch.tensor([[BOS_TOKEN, ANCHOR_TOKEN, ANCHOR_TOKEN]]),
        "token_actors": torch.zeros(1, 3, dtype=torch.long),
        "token_targets": torch.zeros(1, 3, dtype=torch.long),
        "token_timestamps": torch.tensor([[0.0, 60000.0, 120000.0]]),
        "token_item_ids": torch.zeros(1, 3, dtype=torch.long),
        "token_skill_slots": torch.zeros(1, 3, dtype=torch.long),
        "token_monster_types": torch.zeros(1, 3, dtype=torch.long),
        "token_monster_subtypes": torch.zeros(1, 3, dtype=torch.long),
        "token_building_types": torch.zeros(1, 3, dtype=torch.long),
        "token_lane_types": torch.zeros(1, 3, dtype=torch.long),
        "token_tower_types": torch.zeros(1, 3, dtype=torch.long),
        "token_ward_types": torch.zeros(1, 3, dtype=torch.long),
        "anchor_positions": torch.tensor([[1, 2]], dtype=torch.long),
        "anchor_mask": torch.tensor([[True, True]]),
        "event_window_positions": torch.tensor([0], dtype=torch.long),
        "event_window_offsets": torch.tensor([[0, 0]], dtype=torch.long),
        "event_window_counts": torch.tensor([[1, 0]], dtype=torch.long),
        "max_event_window_per_anchor": [1, 0],
        "frame_features": torch.zeros(1, 2, 10, FRAME_FEAT_DIM),
        "anchor_macro_features": torch.zeros(1, 2, MACRO_FEAT_DIM),
        "outcome": torch.tensor([1.0]),
    }
    model = PlanBModel(max_puuids=1)
    model.eval()

    keys_a, minutes_a, blue_win_a = encode_game_keys(model, batch)
    keys_b, minutes_b, blue_win_b = encode_game_keys(model, batch)

    assert torch.equal(minutes_a, minutes_b)
    assert torch.equal(blue_win_a, blue_win_b)
    assert not torch.allclose(keys_a, keys_b), (
        "PlanB eval keys sample z_post; do not claim B=1/B>1 batching "
        "parity without a deterministic eval or RNG-replay contract."
    )


# ---------------------------------------------------------------------------
# Progress logging (bead 1mz) — the skill-aware query builder used to encode
# the whole holdout silently.
# ---------------------------------------------------------------------------


def test_build_holdout_queries_with_meta_emits_progress(capsys, fixture_match_id):
    from model.m4_eval import _build_holdout_queries_with_meta, MID_GAME_MINUTES
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index

    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    model.eval()

    _build_holdout_queries_with_meta(
        model, [fixture_match_id], idx, set(), "cpu", MID_GAME_MINUTES,
        log_every=1,
    )

    out = capsys.readouterr().out
    assert "[m4_eval] skill_aware: encoded 1/1 games" in out, out


def test_build_holdout_queries_with_meta_progress_can_be_disabled(
    capsys, fixture_match_id,
):
    from model.m4_eval import _build_holdout_queries_with_meta, MID_GAME_MINUTES
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index

    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    model.eval()

    _build_holdout_queries_with_meta(
        model, [fixture_match_id], idx, set(), "cpu", MID_GAME_MINUTES,
        log_every=0,
    )

    assert "skill_aware:" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# M4 eval Stage 2 (htw-cfu) — single-pass DataLoader over the holdout.
#
# PROOF BOUNDARY: these tests assert model-forward CALL COUNTS and output
# equivalence. They do NOT measure wall-clock, and no speed-up is claimed here.
# This checkout has no GPU, no plan_b_full_best.pt and 749 local games against
# 51k remote, so a runtime figure measured here would be meaningless.
# ---------------------------------------------------------------------------


class _CountingStubModel:
    """Stand-in for PlanBModel that records how often forward is called.

    Returns correctly-shaped zero h/post_mu — the keys are irrelevant here, the
    call count is the thing under test.
    """

    def __init__(self):
        self.forward_calls = 0
        self.batch_sizes: list[int] = []
        self.training = False

    def eval(self):
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def __call__(self, batch):
        from model.retrieval import D_H, D_Z

        B, T = batch["anchor_mask"].shape
        self.forward_calls += 1
        self.batch_sizes.append(B)
        return {"h": torch.zeros(B, T, D_H),
                "post_mu": torch.zeros(B, T, D_Z),
                "n_anchors": T}


class _RepeatDataset:
    """Serves ``n`` copies of one real MatchDataset sample."""

    def __init__(self, sample, n):
        self.sample = sample
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i >= self.n:
            raise IndexError(i)
        return self.sample


def _patch_repeat_dataset(monkeypatch, fixture_match_id):
    import model.m4_eval as m4
    from model.dataset import MatchDataset, build_puuid_index

    idx = build_puuid_index([fixture_match_id])
    sample = MatchDataset([fixture_match_id], puuid_index=idx)[0]
    monkeypatch.setattr(
        m4, "MatchDataset",
        lambda match_ids, **kwargs: _RepeatDataset(sample, len(match_ids)),
    )
    return sample


def test_build_holdout_queries_combined_cuts_forward_calls_from_2n_to_ceil_n_over_b(
    monkeypatch, fixture_match_id,
):
    """The Stage-2 claim, as a count: 2N model forwards become ceil(N / B).

    2N is the pre-batching cost — ``_build_holdout_queries`` and
    ``_build_holdout_queries_with_meta`` each walked the whole holdout with a
    batch_size=1 DataLoader. Both halves of that ratio are measured here rather
    than asserted from memory: the 2N leg is reproduced by driving the two
    wrappers at batch_size=1.
    """
    import model.m4_eval as m4

    _patch_repeat_dataset(monkeypatch, fixture_match_id)
    N, B = 7, 3
    ids = [f"MATCH-{i}" for i in range(N)]

    # Old cost: two separate B=1 passes over the holdout.
    naive = _CountingStubModel()
    m4._build_holdout_queries(
        naive, ids, {}, set(), "cpu", m4.MID_GAME_MINUTES,
        batch_size=1, log_every=0,
    )
    m4._build_holdout_queries_with_meta(
        naive, ids, {}, set(), "cpu", m4.MID_GAME_MINUTES,
        batch_size=1, log_every=0,
    )
    assert naive.forward_calls == 2 * N == 14
    assert naive.batch_sizes == [1] * (2 * N)

    # New cost: one batched pass yielding queries AND metadata together.
    batched = _CountingStubModel()
    keys, minutes, bands, blue_win, mids = m4._build_holdout_queries_combined(
        batched, ids, {}, set(), "cpu", m4.MID_GAME_MINUTES,
        batch_size=B, log_every=0,
    )

    assert batched.forward_calls == math.ceil(N / B) == 3
    assert batched.batch_sizes == [3, 3, 1]
    # Non-vacuous: the pass actually produced queries.
    Q = keys.shape[0]
    assert Q > 0
    assert minutes.shape[0] == bands.shape[0] == blue_win.shape[0] == Q
    assert len(mids) == Q
    assert set(mids) == set(ids), "every holdout game contributed queries"


def test_build_holdout_queries_wrappers_are_views_of_combined(
    monkeypatch, fixture_match_id,
):
    """Both legacy builders return exactly slices of the combined result.

    Guards the demotion: neither wrapper may drift from the single pass.
    """
    import model.m4_eval as m4

    _patch_repeat_dataset(monkeypatch, fixture_match_id)
    ids = [f"MATCH-{i}" for i in range(5)]
    args = (ids, {}, set(), "cpu", m4.MID_GAME_MINUTES)

    combined = m4._build_holdout_queries_combined(
        _CountingStubModel(), *args, batch_size=2, log_every=0,
    )
    plain = m4._build_holdout_queries(
        _CountingStubModel(), *args, batch_size=2, log_every=0,
    )
    meta = m4._build_holdout_queries_with_meta(
        _CountingStubModel(), *args, batch_size=2, log_every=0,
    )

    assert len(plain) == 2 and len(meta) == 5
    assert torch.equal(plain[0], combined[0])
    assert torch.equal(plain[1], combined[1])
    for got, want in zip(meta[:4], combined[:4]):
        assert torch.equal(got, want)
    assert meta[4] == combined[4]


def test_build_holdout_queries_combined_batch_size_does_not_change_output(
    monkeypatch, fixture_match_id,
):
    """Q, minutes, bands, outcomes and match-id mapping are batch-size invariant.

    Keys are deliberately excluded: the stub model emits zeros, and with a real
    model they are stochastic (see test_encode_game_keys_eval_is_stochastic_
    without_rng_replay). The B=1/B>1 key comparison lives in test_retrieval.py
    under a monkeypatched-deterministic reparameterize.
    """
    import model.m4_eval as m4

    _patch_repeat_dataset(monkeypatch, fixture_match_id)
    ids = [f"MATCH-{i}" for i in range(6)]
    args = (ids, {}, set(), "cpu", m4.MID_GAME_MINUTES)

    ref = m4._build_holdout_queries_combined(
        _CountingStubModel(), *args, batch_size=1, log_every=0,
    )
    for bs in (2, 4, 6, 32):
        got = m4._build_holdout_queries_combined(
            _CountingStubModel(), *args, batch_size=bs, log_every=0,
        )
        assert got[0].shape == ref[0].shape, f"Q changed at batch_size={bs}"
        assert torch.equal(got[1], ref[1]), f"minutes changed at batch_size={bs}"
        assert torch.equal(got[2], ref[2]), f"bands changed at batch_size={bs}"
        assert torch.equal(got[3], ref[3]), f"outcomes changed at batch_size={bs}"
        assert got[4] == ref[4], f"match-id mapping changed at batch_size={bs}"


def test_build_holdout_queries_combined_progress_counts_games_not_batches(
    monkeypatch, capsys, fixture_match_id,
):
    """Progress cadence stays in games even though the loader now yields batches."""
    import model.m4_eval as m4

    _patch_repeat_dataset(monkeypatch, fixture_match_id)
    ids = [f"MATCH-{i}" for i in range(6)]

    m4._build_holdout_queries_combined(
        _CountingStubModel(), ids, {}, set(), "cpu", m4.MID_GAME_MINUTES,
        batch_size=4, log_every=2, log_label="combined",
    )

    out = capsys.readouterr().out
    assert "[m4_eval] combined: encoded 4/6 games" in out, out
    assert "[m4_eval] combined: encoded 6/6 games" in out, out
