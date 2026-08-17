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


# ---------------------------------------------------------------------------
# Progress logging (bead 1mz) — the query loops used to run silent, so a
# multi-hour eval sweep was indistinguishable from a hang.
# ---------------------------------------------------------------------------


def _progress_bundle(N: int = 64, *, with_bands: bool = False):
    from model.retrieval import IndexBundle, KEY_DIM, SCHEMA_VERSION, Whitener
    torch.manual_seed(3)
    corpus_raw = torch.randn(N, KEY_DIM)
    w = Whitener.fit(corpus_raw)
    return IndexBundle(
        corpus_white=w.apply(corpus_raw),
        whitener=w,
        row_match_id=[f"M{i}" for i in range(N)],
        row_anchor_minute=torch.zeros(N, dtype=torch.int64),
        row_blue_win=torch.zeros(N, dtype=torch.int8),
        checkpoint_sha="x", code_sha="y", built_at=0,
        row_rank_band=(
            torch.zeros(N, dtype=torch.int8) if with_bands else None
        ),
        schema_version=SCHEMA_VERSION if with_bands else 1,
    )


def test_query_index_emits_progress_lines(capsys):
    from model.retrieval import KEY_DIM, query_index

    bundle = _progress_bundle()
    q = torch.randn(8, KEY_DIM)
    query_index(bundle, q, k=4, device="cpu", batch_size=2)

    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "query_index:" in ln]
    assert lines, f"expected progress output, got: {out!r}"
    assert "[retrieval] query_index: batch 4/4" in out
    assert "queries=8/8" in out


def test_query_index_single_batch_stays_silent(capsys):
    """The interactive lesson path issues one batch — it must not print."""
    from model.retrieval import KEY_DIM, query_index

    bundle = _progress_bundle()
    q = torch.randn(8, KEY_DIM)
    query_index(bundle, q, k=4, device="cpu", batch_size=256)

    assert "query_index:" not in capsys.readouterr().out


def test_query_index_skill_aware_emits_progress_lines(capsys):
    from model.retrieval import KEY_DIM, query_index_skill_aware

    bundle = _progress_bundle(with_bands=True)
    q = torch.randn(6, KEY_DIM)
    query_index_skill_aware(
        bundle, q, k=4,
        query_rank_bands=torch.zeros(6, dtype=torch.long),
        device="cpu", batch_size=2,
    )

    out = capsys.readouterr().out
    assert "[retrieval] query_index_skill_aware: batch 3/3" in out
    assert "queries=6/6" in out


def test_query_index_skill_aware_single_batch_stays_silent(capsys):
    from model.retrieval import KEY_DIM, query_index_skill_aware

    bundle = _progress_bundle(with_bands=True)
    q = torch.randn(6, KEY_DIM)
    query_index_skill_aware(
        bundle, q, k=4,
        query_rank_bands=torch.zeros(6, dtype=torch.long),
        device="cpu", batch_size=64,
    )

    assert "query_index_skill_aware:" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# M4 eval Stage 2 (htw-cfu) — batched query encoder.
# ---------------------------------------------------------------------------


_ANCHOR_DIM_KEYS = (
    "anchor_positions",
    "frame_features",
    "anchor_macro_features",
    "decision_labels",
)


def _truncate_anchors(sample: dict, n: int) -> dict:
    """A copy of a MatchDataset sample keeping only its first ``n`` anchors.

    Used to get two games with *different* anchor counts out of one fixture, so
    collate has to pad and the batched encoder has to slice by anchor_mask.
    """
    out = dict(sample)
    for key in _ANCHOR_DIM_KEYS:
        out[key] = sample[key][:n]
    for key in sample:
        if key.startswith("next_event_") and key.endswith("_labels"):
            out[key] = sample[key][:n]
    out["window_positions"] = sample["window_positions"][:n]
    return out


def test_encode_game_keys_batched_matches_b1_under_deterministic_reparam(
    monkeypatch, fixture_match_id,
):
    """B>1 batching is exact — but only under a *deterministic* eval forward.

    PlanB's eval forward samples ``z_post``, so raw B=1 vs B>1 keys never match;
    that is exactly what ``test_m4_eval.py::
    test_encode_game_keys_eval_is_stochastic_without_rng_replay`` pins down, and
    it is why this test monkeypatches ``reparameterize`` to its mean instead of
    comparing two stochastic encodings. Do not "simplify" this back into a naive
    parity assertion.

    Even deterministic, parity is *numerical*, not bitwise: float32 GEMM
    reductions order differently at B=1 vs B>1 and the RSSM recurrence carries
    that through ~25 steps, which measures at ~5e-5 max abs on unit-scale keys
    here (mean ~1e-7). The tolerances below are set to catch a real slicing or
    alignment bug — those move keys by O(1), not O(1e-4).
    """
    import model.plan_b_model as plan_b_model_module
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    from model.plan_b_model import PlanBModel
    from model.retrieval import encode_game_keys, encode_game_keys_batched

    monkeypatch.setattr(
        plan_b_model_module, "reparameterize", lambda mu, logvar: mu,
    )

    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    full = ds[0]
    n_full = full["anchor_positions"].shape[0]
    short = _truncate_anchors(full, n_full // 2)
    assert short["anchor_positions"].shape[0] < n_full, "need unequal T to pad"

    model = PlanBModel(max_puuids=len(idx) + 1)
    model.eval()

    samples = [full, short]
    batched = encode_game_keys_batched(model, collate_games(samples))
    assert len(batched) == len(samples)

    for gi, s in enumerate(samples):
        keys_1, minutes_1, blue_win_1 = encode_game_keys(model, collate_games([s]))
        keys_b, minutes_b, blue_win_b = batched[gi]
        # Padded anchors of the short game must be sliced off, not returned.
        assert keys_b.shape == keys_1.shape == (
            s["anchor_positions"].shape[0], keys_1.shape[1],
        )
        diff = (keys_b - keys_1).abs()
        assert diff.max().item() < 2e-4, (
            f"game {gi}: max abs key diff {diff.max().item():.3e} is far above "
            "float32 batched-kernel noise — suspect anchor slicing/alignment"
        )
        assert diff.mean().item() < 1e-5, f"game {gi} mean key diff too large"
        # Minutes and outcome are index arithmetic, so those ARE exact.
        assert torch.equal(minutes_b, minutes_1)
        assert torch.equal(blue_win_b, blue_win_1)


def test_encode_game_keys_batched_identical_rows_encode_identically(
    monkeypatch, fixture_match_id,
):
    """Two *identical* games in one batch encode bitwise-identically.

    This is a determinism check, and unlike the B=1 vs B>1 comparison it needs
    no tolerance: same batch, same kernel, same reduction order.

    It does NOT prove anything about row alignment. Identical inputs make every
    row of ``keys_all`` identical, so any permutation of rows is invisible here
    — a mutation replacing ``keys_all[bi]`` with ``keys_all[0]`` leaves this
    test green. The row-alignment guard is
    ``test_encode_game_keys_batched_keys_belong_to_their_own_row``, which
    batches two genuinely distinct games; do not weaken it into this one.
    """
    import model.plan_b_model as plan_b_model_module
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    from model.plan_b_model import PlanBModel
    from model.retrieval import encode_game_keys_batched

    monkeypatch.setattr(
        plan_b_model_module, "reparameterize", lambda mu, logvar: mu,
    )

    idx = build_puuid_index([fixture_match_id])
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    full = ds[0]
    model = PlanBModel(max_puuids=len(idx) + 1)
    model.eval()

    per_game = encode_game_keys_batched(model, collate_games([full, full]))
    assert torch.equal(per_game[0][0], per_game[1][0])
    assert torch.equal(per_game[0][1], per_game[1][1])


def _first_distinct_match_ids(all_match_ids, count=2):
    """The first ``count`` distinct match_ids in the corpus, else skip.

    Same helper as tests/model/test_baselines_static_only.py — kept local so
    this file has no cross-test-module import.
    """
    out = []
    for mid in all_match_ids:
        if mid not in out:
            out.append(mid)
        if len(out) == count:
            return out
    pytest.skip(f"need at least {count} distinct fixture matches")


def test_encode_game_keys_batched_keys_belong_to_their_own_row(
    monkeypatch, all_match_ids,
):
    """Row ``bi``'s keys come from game ``bi``, not from a neighbouring row.

    This is the cross-row contamination guard, and it is the ONLY test in this
    file that can see one. The rest of the batched-encoder suite batches games
    that are indistinguishable at the key level:

    - ``..._identical_rows_encode_identically`` batches ``[full, full]``.
    - ``..._matches_b1_under_deterministic_reparam`` batches ``[full, short]``
      where ``_truncate_anchors`` keeps the *first* n anchors — a causal prefix
      of the same match, so ``keys_all[0][:n]`` genuinely equals the short
      game's keys.
    - test_m4_eval's batching tests feed N copies of one sample.

    Under all of those, mutating ``keys_all[bi]`` to ``keys_all[0]`` in
    ``encode_game_keys_batched`` is a silent no-op. Two *different* matches
    make it O(1) loud. The distinctness precondition below is asserted, not
    assumed, so this test cannot silently decay into another no-op if the
    fixture corpus ever changes.

    ``reparameterize`` is monkeypatched to its mean for the same reason as the
    B=1 parity test — eval keys sample ``z_post``, so raw re-encodings never
    match. See test_m4_eval.py::
    ``test_encode_game_keys_eval_is_stochastic_without_rng_replay``.
    """
    import model.plan_b_model as plan_b_model_module
    from model.dataset import MatchDataset, build_puuid_index, collate_games
    from model.plan_b_model import PlanBModel
    from model.retrieval import encode_game_keys, encode_game_keys_batched

    monkeypatch.setattr(
        plan_b_model_module, "reparameterize", lambda mu, logvar: mu,
    )

    match_ids = _first_distinct_match_ids(all_match_ids, count=2)
    idx = build_puuid_index(match_ids)
    ds = MatchDataset(match_ids, puuid_index=idx)
    samples = [ds[0], ds[1]]  # ds[i] is match_ids[i] — MatchDataset keeps order

    model = PlanBModel(max_puuids=len(idx) + 1)
    model.eval()

    solo = [encode_game_keys(model, collate_games([s])) for s in samples]
    batched = encode_game_keys_batched(model, collate_games(samples))
    assert len(batched) == len(samples)

    # Precondition: the two games must actually differ over their common
    # anchor prefix. If they did not, a row swap would be undetectable and
    # every assertion below would pass vacuously.
    overlap = min(solo[0][0].shape[0], solo[1][0].shape[0])
    assert overlap > 0, "both fixture games need at least one valid anchor"
    cross = (solo[0][0][:overlap] - solo[1][0][:overlap]).abs().max().item()
    assert cross > 1e-2, (
        f"fixture games {match_ids} encode to near-identical keys "
        f"(max abs diff {cross:.3e}) — this test cannot detect a row swap "
        "against them; pick two genuinely different matches"
    )

    for gi, (keys_1, minutes_1, blue_win_1) in enumerate(solo):
        keys_b, minutes_b, blue_win_b = batched[gi]
        assert keys_b.shape == keys_1.shape, (
            f"row {gi} ({match_ids[gi]}): batched keys have shape "
            f"{tuple(keys_b.shape)}, own-game B=1 keys {tuple(keys_1.shape)}"
        )
        diff = (keys_b - keys_1).abs().max().item()
        # Tolerance is the float32 batched-GEMM noise floor from the B=1
        # parity test (~5e-5). A row swap moves keys by `cross` (O(1e-1)+),
        # three orders of magnitude above this bound.
        assert diff < 2e-4, (
            f"row {gi} ({match_ids[gi]}): max abs key diff {diff:.3e} vs its "
            f"OWN B=1 encoding. Cross-game distance is {cross:.3e}, so this is "
            "row contamination, not float noise — check the keys_all[bi] "
            "indexing in encode_game_keys_batched"
        )
        assert torch.equal(minutes_b, minutes_1), f"row {gi} minutes crossed"
        assert torch.equal(blue_win_b, blue_win_1), f"row {gi} outcome crossed"


def test_encode_game_keys_batched_runs_one_forward_per_batch():
    """The batched encoder calls model.forward exactly once, whatever B is."""
    from model.retrieval import encode_game_keys_batched, D_H, D_Z, KEY_DIM

    calls = []

    class StubModel:
        training = False

        def eval(self):
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def __call__(self, batch):
            B, T = batch["anchor_mask"].shape
            calls.append(B)
            return {"h": torch.zeros(B, T, D_H),
                    "post_mu": torch.zeros(B, T, D_Z),
                    "n_anchors": T}

    B, T = 4, 3
    anchor_mask = torch.ones(B, T, dtype=torch.bool)
    anchor_mask[2, 1:] = False        # a short game
    anchor_mask[3, :] = False         # a game with no valid anchors at all
    batch = {
        "static": torch.zeros(B, 1),
        "players": torch.zeros(B, 10, 1),
        "tokens": torch.zeros(B, 5, dtype=torch.long),
        "anchor_positions": torch.zeros(B, T, dtype=torch.long),
        "anchor_mask": anchor_mask,
        "token_timestamps": torch.full((B, 5), 600000.0),
        "outcome": torch.ones(B),
    }

    per_game = encode_game_keys_batched(StubModel(), batch)

    assert calls == [B], "one forward per batch, not one per game"
    assert len(per_game) == B, "result stays index-aligned with batch rows"
    assert [g[0].shape[0] for g in per_game] == [T, T, 1, 0]
    assert per_game[3][0].shape == (0, KEY_DIM)
    assert per_game[3][1].shape == (0,)
