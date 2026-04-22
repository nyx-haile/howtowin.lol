import math
import os
import random
import time

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


class BucketedBatchSampler(Sampler[list[int]]):
    """Shuffle globally, then batch nearby-cost samples together within buckets."""

    def __init__(
        self,
        sort_keys,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = DEFAULT_BUCKET_SIZE_MULTIPLIER,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.sort_keys = list(sort_keys)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(batch_size, batch_size * max(1, bucket_size_multiplier))
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        n = len(self.sort_keys)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        if not self.sort_keys:
            return

        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        indices = list(range(len(self.sort_keys)))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket.sort(key=self.sort_keys.__getitem__)
            local_batches = [
                bucket[i:i + self.batch_size]
                for i in range(0, len(bucket), self.batch_size)
            ]
            if self.drop_last and local_batches and len(local_batches[-1]) < self.batch_size:
                local_batches.pop()
            if self.shuffle:
                rng.shuffle(local_batches)
            batches.extend(local_batches)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            if batch:
                yield batch


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


def _compute_losses(out, batch):
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
    dec_mask = anchor_mask.unsqueeze(-1).expand_as(dec_targets)
    if dec_mask.any():
        losses["next_decision"] = F.cross_entropy(dec_logits_full[dec_mask], dec_targets[dec_mask], reduction="mean")
    else:
        losses["next_decision"] = dec_logits_full.sum() * 0.0

    mu = out["frame_mu"][:, :-1]
    logvar = out["frame_logvar"][:, :-1]
    target = _frame_delta_labels(batch)
    gauss_nll = 0.5 * (logvar + (target - mu).pow(2) / logvar.exp())
    losses["next_frame"] = _masked_mean(gauss_nll, transition_mask)

    kl = free_bits_kl(
        out["post_mu"].reshape(-1, D_Z), out["post_logvar"].reshape(-1, D_Z),
        out["prior_mu"].reshape(-1, D_Z), out["prior_logvar"].reshape(-1, D_Z),
        free_bits_per_dim=FREE_BITS_PER_DIM,
    ).view_as(anchor_mask)
    losses["kl"] = _masked_mean(kl, anchor_mask)
    return losses


def _rollout_aux_loss(model, out, batch):
    if ROLLOUT_STEPS == 0:
        return torch.tensor(0.0, device=out["event_logits"].device)

    device = out["event_logits"].device
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


def _timed_preflight_step(model, batch, device, use_amp: bool) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    batch = _batch_to_device(batch, device)
    model.zero_grad(set_to_none=True)

    start = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = model(batch)
        losses = _compute_losses(out, batch)
        aux = _rollout_aux_loss(model, out, batch)
        loss = _combined_loss(losses, aux)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    model.zero_grad(set_to_none=True)
    return elapsed


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
        sampled = []
        efficiencies = []
        n_to_sample = min(sample_batches, len(loader))
        for _ in range(n_to_sample):
            batch = next(iterator)
            sampled.append(batch)
            efficiencies.append(_batch_efficiency_metrics(batch))
    finally:
        if sampler_epoch is not None:
            sampler._epoch = sampler_epoch

    if not sampled:
        raise RuntimeError("training preflight could not sample any train batches")

    mean_token_eff = sum(m["token_eff"] for m in efficiencies) / len(efficiencies)
    mean_anchor_eff = sum(m["anchor_eff"] for m in efficiencies) / len(efficiencies)
    mean_recur_eff = sum(m["recur_eff"] for m in efficiencies) / len(efficiencies)

    model.train()
    warm_batch = sampled[0]
    _timed_preflight_step(model, warm_batch, device, use_amp)
    n_timed = min(timed_batches, len(sampled))
    timed = [
        _timed_preflight_step(model, sampled[i], device, use_amp)
        for i in range(n_timed)
    ]
    est_step_seconds = (sum(timed) / max(len(timed), 1)) * PREFLIGHT_STEP_TIME_FUDGE
    projected_epoch_minutes = est_step_seconds * len(loader) / 60.0
    projected_earlystop_hours = projected_epoch_minutes * min(epochs, EARLY_STOP_PATIENCE + 2) / 60.0

    print(
        "[preflight] "
        f"token_eff={mean_token_eff:.3f}  "
        f"anchor_eff={mean_anchor_eff:.3f}  "
        f"recur_eff={mean_recur_eff:.3f}  "
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
        "step_seconds": est_step_seconds,
        "projected_epoch_minutes": projected_epoch_minutes,
        "projected_earlystop_hours": projected_earlystop_hours,
    }


def plan_b_train_loop(train_match_ids, val_match_ids, cold_match_ids,
                      epochs: int = 30, batch_size: int = 8, lr: float = 3e-4,
                      max_puuids: int = 20000,
                      log_every: int = 10,
                      checkpoint_tag: str = "plan_b_full",
                      num_workers: int = 4,
                      run_preflight: bool = True):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    puuid_index = build_puuid_index(train_match_ids, max_puuids=max_puuids)

    exclude = set(val_match_ids) | set(cold_match_ids)
    train_ds = MatchDataset(train_match_ids, puuid_index, exclude_match_ids=exclude)
    val_ds = MatchDataset(val_match_ids, puuid_index, exclude_match_ids=exclude)
    cold_ds = MatchDataset(cold_match_ids, puuid_index, exclude_match_ids=exclude) if cold_match_ids else None

    device = _get_device()
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    model = PlanBModel(max_puuids=max_puuids).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device={device}  amp={use_amp}  params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    history = {"train_loss": [], "game_cold_auc15": [], "player_cold_auc15": []}
    best_cold_auc = -1.0
    patience_left = EARLY_STOP_PATIENCE
    best_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_tag}_best.pt")

    loader_kwargs = dict(
        collate_fn=collate_games,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        pin_memory=use_amp,
        multiprocessing_context=("forkserver" if num_workers > 0 else None),
    )
    if batch_size > 1 and len(train_ds) > batch_size:
        batch_sampler = BucketedBatchSampler(
            estimate_match_batch_sort_keys(train_match_ids),
            batch_size=batch_size,
            shuffle=True,
        )
        print(
            f"bucketed train batching enabled  batch_size={batch_size}  bucket_size={batch_sampler.bucket_size}",
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
        ep_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(loader, start=1):
            batch = _batch_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch)
                losses = _compute_losses(out, batch)
                aux = _rollout_aux_loss(model, out, batch)
                loss = _combined_loss(losses, aux)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.item())
            n_batches += 1
            if log_every > 0 and (step == 1 or step % log_every == 0 or step == n_train_batches):
                elapsed = time.time() - ep_start
                avg_step = elapsed / max(step, 1)
                eta = avg_step * max(n_train_batches - step, 0)
                print(
                    f"epoch {ep+1}/{epochs}  step {step}/{n_train_batches}  "
                    f"loss={loss.item():.4f}  avg_step={avg_step:.2f}s  eta={eta:.1f}s"
                    ,
                    flush=True,
                )
        ep_loss /= max(1, n_batches)
        history["train_loss"].append(ep_loss)

        game_auc = _eval_outcome_auc_at_minute(model, val_ds, device, 15)
        cold_auc = _eval_outcome_auc_at_minute(model, cold_ds, device, 15) if cold_ds else 0.5
        history["game_cold_auc15"].append(game_auc)
        history["player_cold_auc15"].append(cold_auc)

        print(
            f"epoch {ep+1}/{epochs} done  loss={ep_loss:.4f}  "
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
