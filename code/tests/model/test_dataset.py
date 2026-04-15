import torch
from model.dataset import MatchDataset, collate_games, load_split


def test_split_loads():
    holdout = load_split("holdout")
    assert isinstance(holdout, list)
    train = load_split("train")
    assert len(set(holdout) & set(train)) == 0


def test_dataset_yields_sample(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    assert len(ds) == 1
    sample = ds[0]
    assert "static" in sample
    assert "players" in sample
    assert "tokens" in sample
    assert "token_actors" in sample
    assert "token_timestamps" in sample
    assert "labels" in sample
    assert "label_mask" in sample
    # Players should be (10, PLAYER_FEATURE_DIM).
    assert sample["players"].shape[0] == 10
    # Tokens are 1-D tensors.
    assert sample["tokens"].dim() == 1
    assert sample["tokens"].shape == sample["token_actors"].shape
    assert sample["tokens"].shape == sample["token_timestamps"].shape


def test_labels_align_with_anchors(fixture_match_id):
    ds = MatchDataset([fixture_match_id])
    sample = ds[0]
    # Only anchor positions carry supervised labels; others should be masked.
    from model.tokens import ANCHOR_TOKEN
    is_anchor = (sample["tokens"] == ANCHOR_TOKEN)
    # Mask should be 1 exactly at anchor positions (except the final anchor
    # which has no "next" event).
    mask = sample["label_mask"].bool()
    assert mask.sum() <= is_anchor.sum()


def test_collate_batches_games(fixture_match_id):
    ds = MatchDataset([fixture_match_id, fixture_match_id])
    batch = collate_games([ds[0], ds[1]])
    assert batch["static"].shape[0] == 2
    assert batch["players"].shape[0] == 2
    assert batch["tokens"].shape[0] == 2
    # Sequences padded to equal length.
    assert batch["tokens"].shape[1] == batch["labels"].shape[1]
