import torch
import pytest


def _first_distinct_match_ids(all_match_ids, count=2):
    out = []
    for mid in all_match_ids:
        if mid not in out:
            out.append(mid)
        if len(out) == count:
            return out
    pytest.skip(f"need at least {count} distinct fixture matches")


def test_build_static_only_index_returns_one_row_per_anchor(fixture_match_id):
    from model.baselines.static_only_index import build_static_only_index
    from model.plan_b_model import PlanBModel, D_H
    from model.dataset import build_puuid_index, MatchDataset, collate_games
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_static_only_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
    )
    # Same anchor count as the model's index would produce.
    ds = MatchDataset([fixture_match_id], puuid_index=idx)
    batch = collate_games([ds[0]])
    T = batch["anchor_positions"].size(1)
    assert bundle.corpus_white.shape == (T, D_H)
    # All T rows for one game must be identical (static is per-game, not per-anchor).
    first = bundle.corpus_white[0]
    for r in range(1, T):
        assert torch.allclose(bundle.corpus_white[r], first, atol=1e-6)


def test_build_static_only_index_excludes(fixture_match_id):
    from model.baselines.static_only_index import build_static_only_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    idx = build_puuid_index([fixture_match_id])
    model = PlanBModel(max_puuids=len(idx) + 1)
    bundle = build_static_only_index(
        model=model,
        train_match_ids=[fixture_match_id],
        exclude_match_ids={fixture_match_id},
        puuid_index=idx,
        device="cpu",
    )
    assert bundle.corpus_white.shape[0] == 0


def test_build_static_only_index_batched_matches_single_game_order(all_match_ids):
    from model.baselines.static_only_index import build_static_only_index
    from model.plan_b_model import PlanBModel
    from model.dataset import build_puuid_index
    match_ids = _first_distinct_match_ids(all_match_ids, count=2)
    idx = build_puuid_index(match_ids)
    torch.manual_seed(0)
    model = PlanBModel(max_puuids=len(idx) + 1)

    single = build_static_only_index(
        model=model,
        train_match_ids=match_ids,
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
        batch_size=1,
    )
    batched = build_static_only_index(
        model=model,
        train_match_ids=match_ids,
        exclude_match_ids=set(),
        puuid_index=idx,
        device="cpu",
        batch_size=2,
    )

    assert batched.row_match_id == single.row_match_id
    assert torch.equal(batched.row_anchor_minute, single.row_anchor_minute)
    assert torch.equal(batched.row_blue_win, single.row_blue_win)
    assert torch.allclose(batched.corpus_white, single.corpus_white, atol=1e-6)
