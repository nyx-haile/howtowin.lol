import torch

from model.dataset import MACRO_FEAT_DIM, FRAME_FEAT_DIM, MatchDataset, collate_games, load_split
from model.static_features import STATIC_VECTOR_DIM


def test_split_loads():
    holdout = load_split("holdout")
    assert isinstance(holdout, list)
    train = load_split("train")
    assert len(set(holdout) & set(train)) == 0


def test_dataset_yields_sample(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    assert sample["static"].shape == (STATIC_VECTOR_DIM,)
    assert sample["players"].shape[0] == 10
    assert sample["tokens"].dim() == 1
    assert sample["tokens"].shape == sample["token_actors"].shape == sample["token_timestamps"].shape
    assert sample["token_targets"].shape == sample["tokens"].shape
    assert sample["frame_features"].shape[-1] == FRAME_FEAT_DIM
    assert sample["anchor_macro_features"].shape[-1] == MACRO_FEAT_DIM


def test_labels_align_with_anchors(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    from model.tokens import ANCHOR_TOKEN
    is_anchor = sample["tokens"] == ANCHOR_TOKEN
    mask = sample["label_mask"].bool()
    assert mask.sum() <= is_anchor.sum()


def test_richer_anchor_features_present(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    assert sample["frame_features"].shape[-1] > 6
    assert sample["anchor_macro_features"].shape[-1] >= 6
    # current_gold / jungle_cs / kda channels should contribute non-zero signal in a real game.
    assert torch.from_numpy(sample["frame_features"]).abs().sum().item() > 0
    assert torch.from_numpy(sample["anchor_macro_features"]).abs().sum().item() > 0


def test_collate_batches_games(fixture_match_id):
    ds = MatchDataset([fixture_match_id, fixture_match_id])
    batch = collate_games([ds[0], ds[1]])
    assert batch["static"].shape == (2, STATIC_VECTOR_DIM)
    assert batch["players"].shape[0] == 2
    assert batch["tokens"].shape[0] == 2
    assert batch["anchor_mask"].shape[:1] == (2,)
    assert batch["anchor_macro_features"].shape[0] == 2
    assert batch["event_window_positions"].dtype == torch.long
    assert batch["tokens"].shape[1] == batch["labels"].shape[1]
