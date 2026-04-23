import random

import numpy as np
import pytest
import torch

from model.plan_b_train import (
    BucketedBatchSampler,
    SECONDARY_LOSS_SCHEDULE,
    _project_training_runtime,
    _scheduled_time_mask,
    plan_b_train_loop,
    run_training_preflight,
)


def test_bucketed_batch_sampler_reduces_within_batch_cost_span():
    costs = [205, 18, 15, 210, 20, 12, 110, 95]
    sampler = BucketedBatchSampler(costs, batch_size=2, shuffle=False, bucket_size_multiplier=8)
    batches = list(sampler)

    spans = [
        max(costs[i] for i in batch) - min(costs[i] for i in batch)
        for batch in batches
    ]

    assert batches
    assert max(spans) <= 15


def test_secondary_grounding_masks_rotate_by_step():
    mask = torch.ones(1, 4, dtype=torch.bool)

    even = _scheduled_time_mask(
        mask,
        stride=SECONDARY_LOSS_SCHEDULE.decision_stride,
        offset=SECONDARY_LOSS_SCHEDULE.stride_offset(
            epoch=0,
            step=0,
            stride=SECONDARY_LOSS_SCHEDULE.decision_stride,
        ),
    )
    odd = _scheduled_time_mask(
        mask,
        stride=SECONDARY_LOSS_SCHEDULE.decision_stride,
        offset=SECONDARY_LOSS_SCHEDULE.stride_offset(
            epoch=0,
            step=1,
            stride=SECONDARY_LOSS_SCHEDULE.decision_stride,
        ),
    )

    assert even.tolist() == [[True, False, True, False]]
    assert odd.tolist() == [[False, True, False, True]]


def test_rollout_schedule_uses_warmup_and_batch_cadence():
    assert not SECONDARY_LOSS_SCHEDULE.rollout_active(epoch=0, step=0)
    assert SECONDARY_LOSS_SCHEDULE.rollout_active(epoch=1, step=0)
    assert not SECONDARY_LOSS_SCHEDULE.rollout_active(epoch=1, step=1)
    assert SECONDARY_LOSS_SCHEDULE.rollout_step_fraction(epochs=7) == pytest.approx(6 / 7 / 4)


def test_project_training_runtime_accounts_for_sparse_rollout_fraction():
    avg_step, epoch_minutes, earlystop_hours, rollout_fraction = _project_training_runtime(
        base_step_seconds=1.0,
        rollout_step_seconds=1.4,
        steps_per_epoch=60,
        epochs=7,
    )

    assert rollout_fraction == pytest.approx(6 / 7 / 4)
    assert avg_step == pytest.approx(1.0 + 0.4 * rollout_fraction)
    assert epoch_minutes == pytest.approx(avg_step)
    assert earlystop_hours == pytest.approx(avg_step * 7 / 60.0)


def test_overfit_tiny_corpus_drives_loss_down():
    """Smoke test: 5 games x 3 epochs must achieve a lower train loss than its start."""
    from model.dataset import load_split

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    train_ids = load_split("train")[:5]
    val_ids = load_split("holdout")[:2]
    cold_ids = load_split("cold")[:2] if load_split("cold") else []

    history = plan_b_train_loop(
        train_ids, val_ids, cold_ids,
        epochs=3, batch_size=1, lr=1e-3,
        max_puuids=50,
        checkpoint_tag="plan_b_smoke",
        num_workers=0,
        run_preflight=False,
    )
    first = history["train_loss"][0]
    best = min(history["train_loss"][1:], default=first)
    assert best < first, f"smoke test: train loss never beat its start ({first} -> best {best})"


def test_plan_b_train_loop_bucketed_batches_smoke():
    from model.dataset import load_split

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    train_ids = load_split("train")[:4]
    val_ids = load_split("holdout")[:2]
    cold_ids = load_split("cold")[:2] if load_split("cold") else []

    history = plan_b_train_loop(
        train_ids, val_ids, cold_ids,
        epochs=1, batch_size=2, lr=1e-3,
        max_puuids=50,
        checkpoint_tag="plan_b_bucket_smoke",
        num_workers=0,
    )

    assert len(history["train_loss"]) == 1


def test_training_preflight_rejects_impossible_runtime_budget(fixture_match_id):
    from torch.utils.data import DataLoader

    from model.dataset import MatchDataset, build_puuid_index, collate_games
    from model.plan_b_model import PlanBModel

    idx = build_puuid_index([fixture_match_id, fixture_match_id])
    ds = MatchDataset([fixture_match_id, fixture_match_id], puuid_index=idx)
    loader = DataLoader(
        ds,
        batch_sampler=BucketedBatchSampler([1, 1], batch_size=2, shuffle=False, bucket_size_multiplier=1),
        collate_fn=collate_games,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlanBModel(max_puuids=len(idx) + 1).to(device)

    with pytest.raises(RuntimeError, match="training preflight rejected"):
        run_training_preflight(
            model,
            loader,
            device,
            epochs=30,
            use_amp=(device.type == "cuda"),
            sample_batches=1,
            timed_batches=1,
            min_token_efficiency=0.0,
            min_anchor_efficiency=0.0,
            min_recur_efficiency=0.0,
            max_projected_epoch_minutes=0.0,
            max_projected_earlystop_hours=0.0,
        )
