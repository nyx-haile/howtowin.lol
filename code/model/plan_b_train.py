import inspect
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from db import get_conn
from model.dataset import MatchDataset, build_puuid_index, collate_games
from model.plan_b_model import D_Z, PlanBModel
from model.rssm import free_bits_kl
from model.tokens import EVENT_TYPE_TO_ID, event_label_from_type_id

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_checkpoints")
FREE_BITS_PER_DIM = 0.5
KL_WEIGHT_LOSS = 0.05
ROLLOUT_STEPS = 3
ROLLOUT_LOSS_WEIGHTS = [0.05, 0.03, 0.02]
HEAD_WEIGHTS = {"outcome": 0.35, "next_event": 0.35, "next_decision": 0.15, "next_frame": 0.10}
EARLY_STOP_PATIENCE = 5
DEFAULT_BUCKET_SIZE_MULTIPLIER = 64
DEFAULT_BATCH_BUDGET_QUANTILE = 0.60
DEFAULT_BATCH_BUDGET_HEADROOM = 1.15
DEFAULT_CUDA_NUM_WORKERS = 12
DEFAULT_CUDA_PREFETCH_FACTOR = 4
DEFAULT_CPU_NUM_WORKERS = 0
DEFAULT_CPU_PREFETCH_FACTOR = 2
DEFAULT_LOADER_ORDER = "auto"
PREFLIGHT_SAMPLE_BATCHES = 4
PREFLIGHT_TIMED_BATCHES = 2
PREFLIGHT_MIN_TOKEN_EFFICIENCY = 0.80
PREFLIGHT_MIN_ANCHOR_EFFICIENCY = 0.85
PREFLIGHT_MIN_RECUR_SLOT_EFFICIENCY = 0.70
PREFLIGHT_MAX_PROJECTED_EPOCH_MINUTES = 60.0
PREFLIGHT_MAX_PROJECTED_EARLYSTOP_HOURS = 8.0
PREFLIGHT_STEP_TIME_FUDGE = 1.05

ITEM_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["ITEM_PURCHASED"])
SKILL_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["SKILL_LEVEL_UP"])
MONSTER_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["ELITE_MONSTER_KILL"])
BUILDING_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["BUILDING_KILL"])
WARD_PLACED_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["WARD_PLACED"])
WARD_KILL_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["WARD_KILL"])


@dataclass(frozen=True)
class SecondaryLossSchedule:
    """Sparse training schedule for non-primary Plan B objectives.

    Canonical spec priority order:
    - primary: next-event + outcome
    - secondary grounding: next-decision + next-observation
    - rollout: important contract question, but not required every anchor/every batch

    Keep the primary contract dense while making the secondary objectives cheaper.
    """

    decision_stride: int = 2
    frame_stride: int = 2
    rollout_warmup_epochs: int = 1
    rollout_batch_stride: int = 4
    rollout_batch_phase: int = 0

    def stride_offset(self, *, epoch: int, step: int, stride: int) -> int:
        if stride <= 1:
            return 0
        return (epoch + step) % stride

    def rollout_active(self, *, epoch: int, step: int, force: bool | None = None) -> bool:
        if force is False:
            return False
        if ROLLOUT_STEPS <= 0:
            return False
        if force is True:
            return True
        if epoch < self.rollout_warmup_epochs:
            return False
        stride = max(1, self.rollout_batch_stride)
        return ((step - self.rollout_batch_phase) % stride) == 0

    def rollout_step_fraction(self, *, epochs: int) -> float:
        if ROLLOUT_STEPS <= 0:
            return 0.0
        horizon = min(max(epochs, 0), EARLY_STOP_PATIENCE + 2)
        if horizon <= 0:
            return 0.0
        active_epochs = max(horizon - self.rollout_warmup_epochs, 0)
        return (active_epochs / horizon) / max(1, self.rollout_batch_stride)


SECONDARY_LOSS_SCHEDULE = SecondaryLossSchedule()


@dataclass(frozen=True)
class BatchWorkItem:
    token_cost: int
    event_cost: int
    anchor_cost: int
    sort_key: tuple[int, int, int, int]


@dataclass(frozen=True)
class LoaderRuntimeConfig:
    num_workers: int
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory: bool
    multiprocessing_context: str | None
    in_order: bool | None


def _quantile_int(values, q: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(v) for v in values)
    pos = int(round((len(ordered) - 1) * min(max(q, 0.0), 1.0)))
    return ordered[pos]


def _coerce_batch_work_item(value, idx: int) -> BatchWorkItem:
    if isinstance(value, BatchWorkItem):
        return value
    if isinstance(value, (tuple, list)):
        ints = [int(v) for v in value]
        if len(ints) >= 3:
            token_cost = max(ints[0], 1)
            event_cost = max(ints[1], 0)
            anchor_cost = max(ints[2], 1)
            stable = ints[3] if len(ints) > 3 else idx
            return BatchWorkItem(
                token_cost=token_cost,
                event_cost=event_cost,
                anchor_cost=anchor_cost,
                sort_key=(token_cost, event_cost, anchor_cost, stable),
            )
        if len(ints) == 2:
            token_cost = max(ints[0], 1)
            event_cost = max(ints[1], 0)
            return BatchWorkItem(
                token_cost=token_cost,
                event_cost=event_cost,
                anchor_cost=1,
                sort_key=(token_cost, event_cost, 1, idx),
            )
        if len(ints) == 1:
            value = ints[0]
    scalar = max(int(value), 1)
    return BatchWorkItem(
        token_cost=scalar,
        event_cost=scalar,
        anchor_cost=1,
        sort_key=(scalar, scalar, 1, idx),
    )


def _resolve_budget(values, batch_size: int, explicit_budget: int | None) -> int:
    if explicit_budget is not None:
        return max(int(explicit_budget), 1)
    if not values:
        return max(int(batch_size), 1)
    per_item = max(1, _quantile_int(values, DEFAULT_BATCH_BUDGET_QUANTILE))
    return max(int(batch_size), int(math.ceil(per_item * batch_size * DEFAULT_BATCH_BUDGET_HEADROOM)))


class BucketedBatchSampler(Sampler[list[int]]):
    """Shuffle globally, then pack nearby-cost samples into budgeted batches."""

    def __init__(
        self,
        sort_keys,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = DEFAULT_BUCKET_SIZE_MULTIPLIER,
        token_budget: int | None = None,
        event_budget: int | None = None,
        anchor_budget: int | None = None,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.sort_keys = list(sort_keys)
        self.work_items = [
            _coerce_batch_work_item(sort_key, idx)
            for idx, sort_key in enumerate(self.sort_keys)
        ]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(batch_size, batch_size * max(1, bucket_size_multiplier))
        self.token_budget = _resolve_budget(
            [item.token_cost for item in self.work_items],
            batch_size=batch_size,
            explicit_budget=token_budget,
        )
        self.event_budget = _resolve_budget(
            [item.event_cost for item in self.work_items],
            batch_size=batch_size,
            explicit_budget=event_budget,
        )
        self.anchor_budget = _resolve_budget(
            [item.anchor_cost for item in self.work_items],
            batch_size=batch_size,
            explicit_budget=anchor_budget,
        )
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        return len(self._build_batches(self.seed + self._epoch))

    def __iter__(self):
        if not self.work_items:
            return

        batches = self._build_batches(self.seed + self._epoch)
        self._epoch += 1
        for batch in batches:
            if batch:
                yield batch

    def _fits_budget(self, batch, candidate: BatchWorkItem) -> bool:
        if not batch:
            return True
        next_len = len(batch) + 1
        if next_len > self.batch_size:
            return False
        token_sum = sum(item.token_cost for item in batch) + candidate.token_cost
        event_sum = sum(item.event_cost for item in batch) + candidate.event_cost
        anchor_sum = sum(item.anchor_cost for item in batch) + candidate.anchor_cost
        return (
            token_sum <= self.token_budget
            and event_sum <= self.event_budget
            and anchor_sum <= self.anchor_budget
        )

    def _build_batches(self, epoch_seed: int) -> list[list[int]]:
        rng = random.Random(epoch_seed)
        indices = list(range(len(self.work_items)))
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket.sort(key=lambda idx: self.work_items[idx].sort_key)

            local_batches: list[list[int]] = []
            current_batch_indices: list[int] = []
            current_batch_items: list[BatchWorkItem] = []
            for idx in bucket:
                item = self.work_items[idx]
                if current_batch_indices and not self._fits_budget(current_batch_items, item):
                    local_batches.append(current_batch_indices)
                    current_batch_indices = []
                    current_batch_items = []
                current_batch_indices.append(idx)
                current_batch_items.append(item)
            if current_batch_indices:
                local_batches.append(current_batch_indices)

            if self.drop_last:
                local_batches = [
                    batch for batch in local_batches
                    if len(batch) == self.batch_size
                ]
            if self.shuffle:
                rng.shuffle(local_batches)
            batches.extend(local_batches)

        if self.shuffle:
            rng.shuffle(batches)
        return batches


def estimate_match_batch_sort_keys(match_ids):
    """Cheap per-match sort key for bucketed training batches.

    The forward cost is dominated by anchors plus between-anchor events, so
    group matches by approximate token/recurrence work using DB-side counts
    rather than forcing full tokenization up front.
    """
    match_ids = list(match_ids)
    if not match_ids:
        return []

    anchor_counts = {}
    event_counts = {}
    conn = get_conn()
    try:
        chunk_size = 900
        for i in range(0, len(match_ids), chunk_size):
            chunk = match_ids[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)

            frame_rows = conn.execute(
                f"SELECT match_id, COUNT(DISTINCT timestamp_ms) AS n "
                f"FROM frames WHERE match_id IN ({placeholders}) "
                f"AND participant_slot BETWEEN 1 AND 10 GROUP BY match_id",
                chunk,
            ).fetchall()
            for row in frame_rows:
                anchor_counts[row["match_id"]] = int(row["n"] or 0)

            event_rows = conn.execute(
                f"SELECT match_id, COUNT(*) AS n "
                f"FROM events WHERE match_id IN ({placeholders}) "
                f"GROUP BY match_id",
                chunk,
            ).fetchall()
            for row in event_rows:
                event_counts[row["match_id"]] = int(row["n"] or 0)
    finally:
        conn.close()

    sort_keys = []
    for idx, match_id in enumerate(match_ids):
        anchors = anchor_counts.get(match_id, 0)
        events = event_counts.get(match_id, 0)
        est_tokens = anchors + events
        sort_keys.append((est_tokens, events, anchors, idx))
    return sort_keys


def _supports_dataloader_arg(name: str) -> bool:
    return name in inspect.signature(DataLoader).parameters


def _resolve_loader_runtime_config(
    *,
    device: torch.device,
    num_workers: int | None,
    prefetch_factor: int | None,
    persistent_workers: bool | None,
    loader_order: str = DEFAULT_LOADER_ORDER,
) -> LoaderRuntimeConfig:
    loader_order = loader_order.lower()
    if loader_order not in {"auto", "ordered", "out-of-order"}:
        raise ValueError(f"unsupported loader_order={loader_order!r}")

    if num_workers is None:
        default_workers = DEFAULT_CUDA_NUM_WORKERS if device.type == "cuda" else DEFAULT_CPU_NUM_WORKERS
        cpu_cap = max(os.cpu_count() or default_workers, 1)
        num_workers = min(default_workers, cpu_cap)
    num_workers = max(int(num_workers), 0)

    pin_memory = device.type == "cuda"
    if num_workers == 0:
        return LoaderRuntimeConfig(
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
            pin_memory=pin_memory,
            multiprocessing_context=None,
            in_order=None,
        )

    if prefetch_factor is None:
        prefetch_factor = (
            DEFAULT_CUDA_PREFETCH_FACTOR
            if device.type == "cuda" else DEFAULT_CPU_PREFETCH_FACTOR
        )
    prefetch_factor = max(int(prefetch_factor), 1)

    if persistent_workers is None:
        persistent_workers = True

    methods = set(mp.get_all_start_methods())
    if "forkserver" in methods:
        multiprocessing_context = "forkserver"
    elif "spawn" in methods:
        multiprocessing_context = "spawn"
    else:
        multiprocessing_context = None

    in_order = None
    if _supports_dataloader_arg("in_order"):
        if loader_order == "ordered":
            in_order = True
        elif loader_order == "out-of-order":
            in_order = False
        else:
            in_order = not (device.type == "cuda" and num_workers > 0)
    elif loader_order == "out-of-order":
        raise ValueError("this torch build does not support DataLoader(in_order=...)")

    return LoaderRuntimeConfig(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=bool(persistent_workers),
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
        in_order=in_order,
    )


def _build_train_loader_kwargs(
    *,
    device: torch.device,
    num_workers: int | None,
    prefetch_factor: int | None,
    persistent_workers: bool | None,
    loader_order: str = DEFAULT_LOADER_ORDER,
) -> dict:
    config = _resolve_loader_runtime_config(
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        loader_order=loader_order,
    )
    loader_kwargs = dict(
        collate_fn=collate_games,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
        if config.multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = config.multiprocessing_context
    if config.in_order is not None:
        loader_kwargs["in_order"] = config.in_order
    return loader_kwargs


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _outcome_labels(batch):
    win = batch["outcome"]
    T = batch["anchor_positions"].size(1)
    return win.unsqueeze(1).expand(-1, T).float()


def _frame_delta_labels(batch):
    ff = batch["frame_features"]
    return ff[:, 1:] - ff[:, :-1]


def _anchor_mask(batch) -> torch.Tensor:
    return batch["anchor_mask"].bool()


def _transition_mask(batch) -> torch.Tensor:
    mask = _anchor_mask(batch)
    return mask[:, :-1] & mask[:, 1:]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if values.dim() < mask.dim():
        raise ValueError("mask has more dimensions than values")
    expanded = mask
    while expanded.dim() < values.dim():
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    if not expanded.any():
        return values.sum() * 0.0
    return values.masked_select(expanded).mean()


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask], reduction="mean")


def _scheduled_time_mask(mask: torch.Tensor, *, stride: int, offset: int) -> torch.Tensor:
    if stride <= 1 or mask.size(1) == 0:
        return mask
    positions = torch.arange(mask.size(1), device=mask.device)
    keep = (positions % stride) == offset
    return mask & keep.unsqueeze(0)


def _secondary_anchor_mask(
    anchor_mask: torch.Tensor,
    *,
    epoch: int,
    step: int,
    schedule: SecondaryLossSchedule = SECONDARY_LOSS_SCHEDULE,
) -> torch.Tensor:
    return _scheduled_time_mask(
        anchor_mask,
        stride=schedule.decision_stride,
        offset=schedule.stride_offset(epoch=epoch, step=step, stride=schedule.decision_stride),
    )


def _secondary_transition_mask(
    transition_mask: torch.Tensor,
    *,
    epoch: int,
    step: int,
    schedule: SecondaryLossSchedule = SECONDARY_LOSS_SCHEDULE,
) -> torch.Tensor:
    return _scheduled_time_mask(
        transition_mask,
        stride=schedule.frame_stride,
        offset=schedule.stride_offset(epoch=epoch, step=step, stride=schedule.frame_stride),
    )


def _event_loss_components(out, batch):
    event = out["event_factors"]
    anchor_mask = _anchor_mask(batch)
    type_labels = batch["next_event_type_labels"]
    actor_labels = batch["next_event_actor_labels"]
    target_labels = batch["next_event_target_labels"]
    item_labels = batch["next_event_item_labels"]
    skill_labels = batch["next_event_skill_labels"]
    monster_type_labels = batch["next_event_monster_type_labels"]
    monster_subtype_labels = batch["next_event_monster_subtype_labels"]
    building_type_labels = batch["next_event_building_type_labels"]
    lane_type_labels = batch["next_event_lane_type_labels"]
    tower_type_labels = batch["next_event_tower_type_labels"]
    ward_type_labels = batch["next_event_ward_type_labels"]

    losses = []
    losses.append(_masked_ce(event["type_logits"], type_labels, anchor_mask))

    has_event = (type_labels > 0) & anchor_mask
    losses.append(_masked_ce(event["actor_logits"], actor_labels, has_event))
    losses.append(_masked_ce(event["target_logits"], target_labels, has_event))
    losses.append(_masked_ce(event["item_logits"], item_labels, (type_labels == ITEM_EVENT_LABEL) & anchor_mask))
    losses.append(_masked_ce(event["skill_logits"], skill_labels, (type_labels == SKILL_EVENT_LABEL) & anchor_mask))
    losses.append(_masked_ce(event["monster_type_logits"], monster_type_labels, (type_labels == MONSTER_EVENT_LABEL) & anchor_mask))
    losses.append(_masked_ce(event["monster_subtype_logits"], monster_subtype_labels, (type_labels == MONSTER_EVENT_LABEL) & anchor_mask))
    losses.append(_masked_ce(event["building_type_logits"], building_type_labels, (type_labels == BUILDING_EVENT_LABEL) & anchor_mask))
    losses.append(_masked_ce(event["lane_type_logits"], lane_type_labels, (type_labels == BUILDING_EVENT_LABEL) & anchor_mask))
    losses.append(_masked_ce(event["tower_type_logits"], tower_type_labels, (type_labels == BUILDING_EVENT_LABEL) & anchor_mask))
    ward_mask = ((type_labels == WARD_PLACED_EVENT_LABEL) | (type_labels == WARD_KILL_EVENT_LABEL)) & anchor_mask
    losses.append(_masked_ce(event["ward_type_logits"], ward_type_labels, ward_mask))
    return torch.stack(losses).mean()


def _compute_losses(
    out,
    batch,
    *,
    epoch: int = 0,
    step: int = 0,
    schedule: SecondaryLossSchedule = SECONDARY_LOSS_SCHEDULE,
):
    losses = {}
    anchor_mask = _anchor_mask(batch)
    transition_mask = _transition_mask(batch)

    losses["next_event"] = _event_loss_components(out, batch)

    outcome_pred = out["outcome_logits"]
    outcome_targ = _outcome_labels(batch)
    if anchor_mask.any():
        losses["outcome"] = F.binary_cross_entropy_with_logits(outcome_pred[anchor_mask], outcome_targ[anchor_mask])
    else:
        losses["outcome"] = outcome_pred.sum() * 0.0

    dec_logits = out["decision_logits"]
    dec_logits_full = F.pad(dec_logits, (0, 1))
    dec_targets = batch["decision_labels"].long()
    decision_anchor_mask = _secondary_anchor_mask(anchor_mask, epoch=epoch, step=step, schedule=schedule)
    dec_mask = decision_anchor_mask.unsqueeze(-1).expand_as(dec_targets)
    if dec_mask.any():
        losses["next_decision"] = F.cross_entropy(dec_logits_full[dec_mask], dec_targets[dec_mask], reduction="mean")
    else:
        losses["next_decision"] = dec_logits_full.sum() * 0.0

    mu = out["frame_mu"][:, :-1]
    logvar = out["frame_logvar"][:, :-1]
    target = _frame_delta_labels(batch)
    gauss_nll = 0.5 * (logvar + (target - mu).pow(2) / logvar.exp())
    scheduled_transition_mask = _secondary_transition_mask(
        transition_mask, epoch=epoch, step=step, schedule=schedule
    )
    losses["next_frame"] = _masked_mean(gauss_nll, scheduled_transition_mask)

    kl = free_bits_kl(
        out["post_mu"].reshape(-1, D_Z), out["post_logvar"].reshape(-1, D_Z),
        out["prior_mu"].reshape(-1, D_Z), out["prior_logvar"].reshape(-1, D_Z),
        free_bits_per_dim=FREE_BITS_PER_DIM,
    ).view_as(anchor_mask)
    losses["kl"] = _masked_mean(kl, anchor_mask)
    return losses


def _rollout_aux_loss(
    model,
    out,
    batch,
    *,
    epoch: int = 0,
    step: int = 0,
    schedule: SecondaryLossSchedule = SECONDARY_LOSS_SCHEDULE,
    force: bool | None = None,
):
    if ROLLOUT_STEPS == 0:
        return torch.tensor(0.0, device=out["outcome_logits"].device)
    if not schedule.rollout_active(epoch=epoch, step=step, force=force):
        return torch.tensor(0.0, device=out["outcome_logits"].device)

    device = out["outcome_logits"].device
    anchor_mask = _anchor_mask(batch)
    total = torch.tensor(0.0, device=device)
    weight_total = 0.0

    for b in range(anchor_mask.size(0)):
        n_anchors = int(anchor_mask[b].sum().item())
        if n_anchors < 2 + ROLLOUT_STEPS:
            continue

        seed_t = n_anchors - ROLLOUT_STEPS - 1
        rollout_steps = model.rollout_prior_single(
            h0=out["h"][b, seed_t],
            z0=out["z"][b, seed_t],
            static_tokens=out["static_tokens"][b],
            token_embeddings=out["token_embeddings"][b],
            event_window_positions=batch["event_window_positions"],
            event_window_offsets=batch["event_window_offsets"][b],
            event_window_counts=batch["event_window_counts"][b],
            anchor_mask=anchor_mask[b],
            start_anchor=seed_t,
            n_steps=ROLLOUT_STEPS,
        )

        for step_idx, step_out in enumerate(rollout_steps):
            target_t = seed_t + step_idx + 1
            decoded = model.decode_anchor_repr(step_out["anchor_repr"])
            event_type_logits = decoded["event_factors"]["type_logits"]
            outcome_logits = decoded["outcome_logits"]
            true_event = batch["next_event_type_labels"][b:b + 1, target_t]
            true_outcome = batch["outcome"][b:b + 1]
            weight = ROLLOUT_LOSS_WEIGHTS[step_idx]
            total = total + weight * (
                F.cross_entropy(event_type_logits, true_event)
                + F.binary_cross_entropy_with_logits(outcome_logits, true_outcome)
            )
            weight_total += weight

    if weight_total == 0.0:
        return total
    return total / weight_total


def _combined_loss(losses, rollout_aux):
    total = (
        HEAD_WEIGHTS["next_event"] * losses["next_event"]
        + HEAD_WEIGHTS["outcome"] * losses["outcome"]
        + HEAD_WEIGHTS["next_decision"] * losses["next_decision"]
        + HEAD_WEIGHTS["next_frame"] * losses["next_frame"]
        + KL_WEIGHT_LOSS * losses["kl"]
        + rollout_aux
    )
    return total


def _project_training_runtime(
    *,
    base_step_seconds: float,
    rollout_step_seconds: float | None,
    steps_per_epoch: int,
    epochs: int,
    schedule: SecondaryLossSchedule = SECONDARY_LOSS_SCHEDULE,
):
    rollout_fraction = schedule.rollout_step_fraction(epochs=epochs)
    rollout_extra = 0.0
    if rollout_step_seconds is not None:
        rollout_extra = max(rollout_step_seconds - base_step_seconds, 0.0)
    avg_step_seconds = base_step_seconds + rollout_fraction * rollout_extra
    projected_epoch_minutes = avg_step_seconds * steps_per_epoch / 60.0
    projected_earlystop_hours = (
        projected_epoch_minutes * min(max(epochs, 0), EARLY_STOP_PATIENCE + 2) / 60.0
    )
    return avg_step_seconds, projected_epoch_minutes, projected_earlystop_hours, rollout_fraction


@torch.no_grad()
def _eval_outcome_auc_at_minute(model, ds, device, target_minute: int = 15) -> float:
    from sklearn.metrics import roc_auc_score

    loader = DataLoader(ds, batch_size=8, collate_fn=collate_games, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    y_true, y_score = [], []
    model.eval()
    for batch in loader:
        batch = _batch_to_device(batch, device)
        out = model(batch)
        if target_minute >= out["outcome_logits"].size(1):
            continue
        valid = batch["anchor_mask"][:, target_minute].bool()
        if not valid.any():
            continue
        probs = torch.sigmoid(out["outcome_logits"][valid, target_minute])
        y_score.extend(probs.cpu().tolist())
        y_true.extend(batch["outcome"][valid].cpu().tolist())
    model.train()
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _batch_efficiency_metrics(batch):
    batch_size = batch["tokens"].size(0)

    token_counts = (~batch["key_pad_mask"]).sum(dim=1, dtype=torch.float32)
    token_slots = max(batch["tokens"].numel(), 1)
    token_eff = float(token_counts.sum().item() / token_slots)

    anchor_counts = batch["anchor_mask"].sum(dim=1, dtype=torch.float32)
    anchor_slots = max(batch["anchor_mask"].numel(), 1)
    anchor_eff = float(anchor_counts.sum().item() / anchor_slots)

    useful_events = int(batch["event_window_counts"].sum().item())
    per_anchor_max = batch["event_window_counts"].amax(dim=0)
    recur_slots = batch_size * max(int(per_anchor_max.sum().item()), 1)
    recur_eff = float(useful_events / recur_slots)

    return {
        "token_eff": token_eff,
        "anchor_eff": anchor_eff,
        "recur_eff": recur_eff,
    }


def _run_preflight_loaded_batch(
    model,
    batch,
    device,
    use_amp: bool,
    *,
    epoch: int = 0,
    step: int = 0,
    force_rollout: bool | None = None,
):
    batch = _batch_to_device(batch, device)
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = model(batch)
        losses = _compute_losses(out, batch, epoch=epoch, step=step)
        aux = _rollout_aux_loss(model, out, batch, epoch=epoch, step=step, force=force_rollout)
        loss = _combined_loss(losses, aux)
    loss.backward()

    model.zero_grad(set_to_none=True)


def _timed_preflight_iteration(
    model,
    iterator,
    device,
    use_amp: bool,
    *,
    epoch: int = 0,
    step: int = 0,
    force_rollout: bool | None = None,
):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    batch = next(iterator)
    _run_preflight_loaded_batch(
        model,
        batch,
        device,
        use_amp,
        epoch=epoch,
        step=step,
        force_rollout=force_rollout,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start, batch


def _measure_preflight_pass(
    model,
    loader,
    device,
    use_amp: bool,
    *,
    timed_batches: int,
    epoch: int = 0,
    force_rollout: bool | None = None,
):
    available = len(loader)
    if available <= 0:
        raise RuntimeError("training preflight could not sample any train batches")

    iterator = iter(loader)
    if available == 1:
        elapsed, _ = _timed_preflight_iteration(
            model,
            iterator,
            device,
            use_amp,
            epoch=epoch,
            step=0,
            force_rollout=force_rollout,
        )
        return [elapsed]

    _timed_preflight_iteration(
        model,
        iterator,
        device,
        use_amp,
        epoch=epoch,
        step=0,
        force_rollout=force_rollout,
    )
    n_timed = max(1, min(timed_batches, available - 1))
    timed = []
    for step_idx in range(n_timed):
        elapsed, _ = _timed_preflight_iteration(
            model,
            iterator,
            device,
            use_amp,
            epoch=epoch,
            step=step_idx + 1,
            force_rollout=force_rollout,
        )
        timed.append(elapsed)
    return timed


def run_training_preflight(
    model,
    loader,
    device,
    *,
    epochs: int,
    use_amp: bool,
    sample_batches: int = PREFLIGHT_SAMPLE_BATCHES,
    timed_batches: int = PREFLIGHT_TIMED_BATCHES,
    min_token_efficiency: float = PREFLIGHT_MIN_TOKEN_EFFICIENCY,
    min_anchor_efficiency: float = PREFLIGHT_MIN_ANCHOR_EFFICIENCY,
    min_recur_efficiency: float = PREFLIGHT_MIN_RECUR_SLOT_EFFICIENCY,
    max_projected_epoch_minutes: float = PREFLIGHT_MAX_PROJECTED_EPOCH_MINUTES,
    max_projected_earlystop_hours: float = PREFLIGHT_MAX_PROJECTED_EARLYSTOP_HOURS,
):
    sampler = getattr(loader, "batch_sampler", None)
    sampler_epoch = getattr(sampler, "_epoch", None)

    try:
        iterator = iter(loader)
        efficiencies = []
        n_to_sample = min(sample_batches, len(loader))
        for _ in range(n_to_sample):
            batch = next(iterator)
            efficiencies.append(_batch_efficiency_metrics(batch))

        if not efficiencies:
            raise RuntimeError("training preflight could not sample any train batches")

        mean_token_eff = sum(m["token_eff"] for m in efficiencies) / len(efficiencies)
        mean_anchor_eff = sum(m["anchor_eff"] for m in efficiencies) / len(efficiencies)
        mean_recur_eff = sum(m["recur_eff"] for m in efficiencies) / len(efficiencies)

        model.train()
        base_timed = _measure_preflight_pass(
            model,
            loader,
            device,
            use_amp,
            timed_batches=timed_batches,
            epoch=0,
            force_rollout=False,
        )
        base_step_seconds = (sum(base_timed) / len(base_timed)) * PREFLIGHT_STEP_TIME_FUDGE

        rollout_step_seconds = None
        rollout_fraction = SECONDARY_LOSS_SCHEDULE.rollout_step_fraction(epochs=epochs)
        if rollout_fraction > 0.0:
            rollout_epoch = SECONDARY_LOSS_SCHEDULE.rollout_warmup_epochs
            rollout_timed = _measure_preflight_pass(
                model,
                loader,
                device,
                use_amp,
                timed_batches=timed_batches,
                epoch=rollout_epoch,
                force_rollout=True,
            )
            rollout_step_seconds = (sum(rollout_timed) / len(rollout_timed)) * PREFLIGHT_STEP_TIME_FUDGE

        est_step_seconds, projected_epoch_minutes, projected_earlystop_hours, rollout_fraction = _project_training_runtime(
            base_step_seconds=base_step_seconds,
            rollout_step_seconds=rollout_step_seconds,
            steps_per_epoch=len(loader),
            epochs=epochs,
        )
    finally:
        if sampler_epoch is not None:
            sampler._epoch = sampler_epoch

    print(
        "[preflight/e2e] "
        f"token_eff={mean_token_eff:.3f}  "
        f"anchor_eff={mean_anchor_eff:.3f}  "
        f"recur_eff={mean_recur_eff:.3f}  "
        f"base={base_step_seconds:.2f}s  "
        + (
            f"rollout={rollout_step_seconds:.2f}s  "
            if rollout_step_seconds is not None else
            "rollout=off  "
        )
        + f"rollout_frac={rollout_fraction:.3f}  "
        f"step={est_step_seconds:.2f}s  "
        f"epoch≈{projected_epoch_minutes:.1f}m  "
        f"earlystop≈{projected_earlystop_hours:.1f}h",
        flush=True,
    )

    failures = []
    if mean_token_eff < min_token_efficiency:
        failures.append(
            f"token batching efficiency {mean_token_eff:.3f} < {min_token_efficiency:.3f}"
        )
    if mean_anchor_eff < min_anchor_efficiency:
        failures.append(
            f"anchor batching efficiency {mean_anchor_eff:.3f} < {min_anchor_efficiency:.3f}"
        )
    if mean_recur_eff < min_recur_efficiency:
        failures.append(
            f"event-recurrence efficiency {mean_recur_eff:.3f} < {min_recur_efficiency:.3f}"
        )
    if projected_epoch_minutes > max_projected_epoch_minutes:
        failures.append(
            f"projected epoch time {projected_epoch_minutes:.1f}m > {max_projected_epoch_minutes:.1f}m"
        )
    if projected_earlystop_hours > max_projected_earlystop_hours:
        failures.append(
            f"projected early-stop time {projected_earlystop_hours:.1f}h > {max_projected_earlystop_hours:.1f}h"
        )

    if failures:
        raise RuntimeError(
            "training preflight rejected this run: " + "; ".join(failures)
        )

    return {
        "token_eff": mean_token_eff,
        "anchor_eff": mean_anchor_eff,
        "recur_eff": mean_recur_eff,
        "base_step_seconds": base_step_seconds,
        "rollout_step_seconds": rollout_step_seconds,
        "rollout_step_fraction": rollout_fraction,
        "step_seconds": est_step_seconds,
        "projected_epoch_minutes": projected_epoch_minutes,
        "projected_earlystop_hours": projected_earlystop_hours,
    }


def plan_b_train_loop(train_match_ids, val_match_ids, cold_match_ids,
                      epochs: int = 30, batch_size: int = 8, lr: float = 3e-4,
                      max_puuids: int = 20000,
                      log_every: int = 10,
                      checkpoint_tag: str = "plan_b_full",
                      num_workers: int | None = None,
                      prefetch_factor: int | None = None,
                      persistent_workers: bool | None = None,
                      loader_order: str = DEFAULT_LOADER_ORDER,
                      train_cache_size: int | None = None,
                      run_preflight: bool = True):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    puuid_index = build_puuid_index(train_match_ids, max_puuids=max_puuids)

    exclude = set(val_match_ids) | set(cold_match_ids)
    device = _get_device()
    use_amp = device.type == "cuda"
    loader_config = _resolve_loader_runtime_config(
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        loader_order=loader_order,
    )
    if train_cache_size is None:
        train_cache_size = 0 if loader_config.num_workers > 0 else 8000

    train_ds = MatchDataset(
        train_match_ids,
        puuid_index,
        exclude_match_ids=exclude,
        cache_size=max(int(train_cache_size), 0),
    )
    val_ds = MatchDataset(val_match_ids, puuid_index, exclude_match_ids=exclude)
    cold_ds = MatchDataset(cold_match_ids, puuid_index, exclude_match_ids=exclude) if cold_match_ids else None

    if use_amp:
        torch.backends.cudnn.benchmark = True
    model = PlanBModel(max_puuids=max_puuids).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device={device}  amp={use_amp}  params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(
        "secondary loss schedule  "
        f"decision_stride={SECONDARY_LOSS_SCHEDULE.decision_stride}  "
        f"frame_stride={SECONDARY_LOSS_SCHEDULE.frame_stride}  "
        f"rollout_warmup_epochs={SECONDARY_LOSS_SCHEDULE.rollout_warmup_epochs}  "
        f"rollout_batch_stride={SECONDARY_LOSS_SCHEDULE.rollout_batch_stride}",
        flush=True,
    )
    loader_kwargs = _build_train_loader_kwargs(
        device=device,
        num_workers=loader_config.num_workers,
        prefetch_factor=loader_config.prefetch_factor,
        persistent_workers=loader_config.persistent_workers,
        loader_order=loader_order,
    )
    print(
        "train loader  "
        f"num_workers={loader_config.num_workers}  "
        f"prefetch_factor={loader_config.prefetch_factor or 0}  "
        f"persistent_workers={loader_config.persistent_workers}  "
        f"pin_memory={loader_config.pin_memory}  "
        + (
            f"in_order={loader_config.in_order}  "
            if loader_config.in_order is not None else
            ""
        )
        + f"train_cache_size={max(int(train_cache_size), 0)}",
        flush=True,
    )

    history = {
        "train_loss": [],
        "train_total_loss": [],
        "game_cold_auc15": [],
        "player_cold_auc15": [],
    }
    best_cold_auc = -1.0
    patience_left = EARLY_STOP_PATIENCE
    best_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_tag}_best.pt")

    if batch_size > 1 and len(train_ds) > batch_size:
        batch_sampler = BucketedBatchSampler(
            estimate_match_batch_sort_keys(train_match_ids),
            batch_size=batch_size,
            shuffle=True,
        )
        print(
            "budgeted train batching enabled  "
            f"max_batch_size={batch_size}  "
            f"bucket_size={batch_sampler.bucket_size}  "
            f"token_budget={batch_sampler.token_budget}  "
            f"event_budget={batch_sampler.event_budget}  "
            f"anchor_budget={batch_sampler.anchor_budget}",
            flush=True,
        )
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    n_train_batches = len(loader)

    if run_preflight:
        run_training_preflight(
            model,
            loader,
            device,
            epochs=epochs,
            use_amp=use_amp,
        )

    for ep in range(epochs):
        model.train()
        ep_start = time.time()
        ep_core_loss = 0.0
        ep_total_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(loader, start=1):
            train_step = step - 1
            batch = _batch_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch)
                losses = _compute_losses(out, batch, epoch=ep, step=train_step)
                aux = _rollout_aux_loss(model, out, batch, epoch=ep, step=train_step)
                core_loss = _combined_loss(losses, 0.0)
                loss = _combined_loss(losses, aux)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_core_loss += float(core_loss.item())
            ep_total_loss += float(loss.item())
            n_batches += 1
            if log_every > 0 and (step == 1 or step % log_every == 0 or step == n_train_batches):
                elapsed = time.time() - ep_start
                avg_step = elapsed / max(step, 1)
                eta = avg_step * max(n_train_batches - step, 0)
                print(
                    f"epoch {ep+1}/{epochs}  step {step}/{n_train_batches}  "
                    f"core_loss={core_loss.item():.4f}  total_loss={loss.item():.4f}  "
                    f"avg_step={avg_step:.2f}s  eta={eta:.1f}s"
                    ,
                    flush=True,
                )
        ep_core_loss /= max(1, n_batches)
        ep_total_loss /= max(1, n_batches)
        history["train_loss"].append(ep_core_loss)
        history["train_total_loss"].append(ep_total_loss)

        game_auc = _eval_outcome_auc_at_minute(model, val_ds, device, 15)
        cold_auc = _eval_outcome_auc_at_minute(model, cold_ds, device, 15) if cold_ds else 0.5
        history["game_cold_auc15"].append(game_auc)
        history["player_cold_auc15"].append(cold_auc)

        print(
            f"epoch {ep+1}/{epochs} done  core_loss={ep_core_loss:.4f}  "
            f"total_loss={ep_total_loss:.4f}  "
            f"game_cold_auc15={game_auc:.3f}  player_cold_auc15={cold_auc:.3f}",
            flush=True,
        )

        if cold_auc > best_cold_auc + 1e-4:
            best_cold_auc = cold_auc
            patience_left = EARLY_STOP_PATIENCE
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "max_puuids": max_puuids,
                "history": history,
                "loss_config": {
                    "head_weights": dict(HEAD_WEIGHTS),
                    "kl_weight_loss": KL_WEIGHT_LOSS,
                    "free_bits_per_dim": FREE_BITS_PER_DIM,
                    "rollout_steps": ROLLOUT_STEPS,
                    "rollout_loss_weights": list(ROLLOUT_LOSS_WEIGHTS),
                    "secondary_schedule": asdict(SECONDARY_LOSS_SCHEDULE),
                },
                "runtime_config": {
                    "num_workers": loader_config.num_workers,
                    "prefetch_factor": loader_config.prefetch_factor,
                    "persistent_workers": loader_config.persistent_workers,
                    "pin_memory": loader_config.pin_memory,
                    "multiprocessing_context": loader_config.multiprocessing_context,
                    "in_order": loader_config.in_order,
                    "train_cache_size": max(int(train_cache_size), 0),
                },
            }, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"Early stop at epoch {ep+1} "
                    f"(no player-cold AUC@15 gain for {EARLY_STOP_PATIENCE} epochs).",
                    flush=True,
                )
                break

    return history
