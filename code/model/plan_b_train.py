import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, build_puuid_index, collate_games
from model.plan_b_model import D_Z, PlanBModel
from model.rollout import rollout_prior
from model.rssm import free_bits_kl
from model.tokens import EVENT_TYPE_TO_ID, event_label_from_type_id

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_checkpoints")
FREE_BITS_PER_DIM = 0.5
KL_WEIGHT_LOSS = 0.05
ROLLOUT_STEPS = 3
ROLLOUT_LOSS_WEIGHTS = [0.05, 0.03, 0.02]
HEAD_WEIGHTS = {"outcome": 0.35, "next_event": 0.35, "next_decision": 0.15, "next_frame": 0.10}
EARLY_STOP_PATIENCE = 5

ITEM_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["ITEM_PURCHASED"])
SKILL_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["SKILL_LEVEL_UP"])
MONSTER_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["ELITE_MONSTER_KILL"])
BUILDING_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["BUILDING_KILL"])
WARD_PLACED_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["WARD_PLACED"])
WARD_KILL_EVENT_LABEL = event_label_from_type_id(EVENT_TYPE_TO_ID["WARD_KILL"])


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


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask], reduction="mean")


def _event_loss_components(out, batch):
    event = out["event_factors"]
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
    losses.append(F.cross_entropy(event["type_logits"].reshape(-1, event["type_logits"].size(-1)), type_labels.reshape(-1)))

    has_event = type_labels > 0
    losses.append(_masked_ce(event["actor_logits"], actor_labels, has_event))
    losses.append(_masked_ce(event["target_logits"], target_labels, has_event))
    losses.append(_masked_ce(event["item_logits"], item_labels, type_labels == ITEM_EVENT_LABEL))
    losses.append(_masked_ce(event["skill_logits"], skill_labels, type_labels == SKILL_EVENT_LABEL))
    losses.append(_masked_ce(event["monster_type_logits"], monster_type_labels, type_labels == MONSTER_EVENT_LABEL))
    losses.append(_masked_ce(event["monster_subtype_logits"], monster_subtype_labels, type_labels == MONSTER_EVENT_LABEL))
    losses.append(_masked_ce(event["building_type_logits"], building_type_labels, type_labels == BUILDING_EVENT_LABEL))
    losses.append(_masked_ce(event["lane_type_logits"], lane_type_labels, type_labels == BUILDING_EVENT_LABEL))
    losses.append(_masked_ce(event["tower_type_logits"], tower_type_labels, type_labels == BUILDING_EVENT_LABEL))
    ward_mask = (type_labels == WARD_PLACED_EVENT_LABEL) | (type_labels == WARD_KILL_EVENT_LABEL)
    losses.append(_masked_ce(event["ward_type_logits"], ward_type_labels, ward_mask))
    return torch.stack(losses).mean()


def _compute_losses(out, batch):
    losses = {}
    losses["next_event"] = _event_loss_components(out, batch)

    outcome_pred = out["outcome_logits"]
    outcome_targ = _outcome_labels(batch)
    losses["outcome"] = F.binary_cross_entropy_with_logits(outcome_pred, outcome_targ)

    dec_logits = out["decision_logits"]
    dec_logits_full = F.pad(dec_logits, (0, 1))
    losses["next_decision"] = F.cross_entropy(dec_logits_full.view(-1, 7), batch["decision_labels"].view(-1).long())

    mu = out["frame_mu"][:, :-1]
    logvar = out["frame_logvar"][:, :-1]
    target = _frame_delta_labels(batch)
    gauss_nll = 0.5 * (logvar + (target - mu).pow(2) / logvar.exp())
    losses["next_frame"] = gauss_nll.mean()

    kl = free_bits_kl(
        out["post_mu"].reshape(-1, D_Z), out["post_logvar"].reshape(-1, D_Z),
        out["prior_mu"].reshape(-1, D_Z), out["prior_logvar"].reshape(-1, D_Z),
        free_bits_per_dim=FREE_BITS_PER_DIM,
    )
    losses["kl"] = kl.mean()
    return losses


def _rollout_aux_loss(model, out, batch):
    if ROLLOUT_STEPS == 0:
        return torch.tensor(0.0, device=out["event_logits"].device)
    steps = rollout_prior(model.rssm, out["h_final"], out["z_final"], n_steps=ROLLOUT_STEPS, action_summary=None)
    last_type = batch["next_event_type_labels"][:, -1]
    last_outcome = _outcome_labels(batch)[:, -1]
    total = 0.0
    for (_h, z, _mu, _lv), w in zip(steps, ROLLOUT_LOSS_WEIGHTS):
        ev = model.head_event(z)
        oc_logits = model.head_outcome(z)
        total = total + w * (
            F.cross_entropy(ev["type_logits"], last_type) +
            F.binary_cross_entropy_with_logits(oc_logits, last_outcome)
        )
    return total


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
        T = out["n_anchors"]
        minute_idx = min(target_minute, T - 1)
        probs = torch.sigmoid(out["outcome_logits"])[:, minute_idx]
        y_score.extend(probs.cpu().tolist())
        y_true.extend(batch["outcome"].cpu().tolist())
    model.train()
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def plan_b_train_loop(train_match_ids, val_match_ids, cold_match_ids,
                      epochs: int = 30, batch_size: int = 8, lr: float = 3e-4,
                      max_puuids: int = 20000,
                      log_every: int = 10,
                      checkpoint_tag: str = "plan_b_full",
                      num_workers: int = 4):
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
    print(f"device={device}  amp={use_amp}  params={sum(p.numel() for p in model.parameters()):,}")

    history = {"train_loss": [], "game_cold_auc15": [], "player_cold_auc15": []}
    best_cold_auc = -1.0
    patience_left = EARLY_STOP_PATIENCE
    best_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_tag}_best.pt")

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_games,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        pin_memory=use_amp,
        multiprocessing_context=("forkserver" if num_workers > 0 else None),
    )
    n_train_batches = len(loader)

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
                )
        ep_loss /= max(1, n_batches)
        history["train_loss"].append(ep_loss)

        game_auc = _eval_outcome_auc_at_minute(model, val_ds, device, 15)
        cold_auc = _eval_outcome_auc_at_minute(model, cold_ds, device, 15) if cold_ds else 0.5
        history["game_cold_auc15"].append(game_auc)
        history["player_cold_auc15"].append(cold_auc)

        print(f"epoch {ep+1}/{epochs} done  loss={ep_loss:.4f}  game_cold_auc15={game_auc:.3f}  player_cold_auc15={cold_auc:.3f}")

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
                print(f"Early stop at epoch {ep+1} (no player-cold AUC@15 gain for {EARLY_STOP_PATIENCE} epochs).")
                break

    return history
