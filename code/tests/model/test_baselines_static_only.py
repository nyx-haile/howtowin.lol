import torch


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
