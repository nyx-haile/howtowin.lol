import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, build_puuid_index, collate_games, load_split
from model.plan_b_model import PlanBModel, D_Z
from model.rssm import free_bits_kl
from model.rollout import rollout_prior
from model.tokens import NUM_EVENT_TYPES


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_checkpoints")
FREE_BITS_PER_DIM = 0.5
KL_WEIGHT_LOSS = 0.05
ROLLOUT_STEPS = 3
ROLLOUT_LOSS_WEIGHTS = [0.05, 0.03, 0.02]
HEAD_WEIGHTS = {"outcome": 0.35, "next_event": 0.35,
                "next_decision": 0.15, "next_frame": 0.10}
EARLY_STOP_PATIENCE = 5


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _outcome_labels(batch):
    # (B,) -> broadcast to (B, T)
    win = batch["outcome"]  # dataset must supply this (game winner label per sample)
    T = batch["anchor_positions"].size(1)
    return win.unsqueeze(1).expand(-1, T).float()


def _frame_delta_labels(batch):
    # Delta of frame_features between consecutive anchors: (B, T-1, 10, 6)
    ff = batch["frame_features"]
    return ff[:, 1:] - ff[:, :-1]


def _compute_losses(out, batch):
    losses = {}

    event_logits = out["event_logits"]  # (B, T, NUM_EVENT_TYPES)
    # Need to gather labels at anchor positions from the full-sequence labels
    anchor_pos = batch["anchor_positions"]  # (B, T)
    B, T = anchor_pos.shape
    # labels is (B, L, NUM_EVENT_TYPES) — gather at anchor positions
    idx = anchor_pos.unsqueeze(-1).expand(B, T, out["event_logits"].size(-1))
    anchor_labels = torch.gather(batch["labels"], 1, idx)  # (B, T, NUM_EVENT_TYPES)
    losses["next_event"] = F.binary_cross_entropy_with_logits(
        event_logits, anchor_labels.float(), reduction="mean"
    )

    outcome_pred = out["outcome_logits"]  # (B, T)
    outcome_targ = _outcome_labels(batch)
    losses["outcome"] = F.binary_cross_entropy_with_logits(outcome_pred, outcome_targ)

    dec_logits = out["decision_logits"]  # (B, T, 10, 6)
    # Append a no-decision slot as zero-logit.
    no_dec_col = torch.zeros(*dec_logits.shape[:-1], 1, device=dec_logits.device)
    dec_logits_full = torch.cat([dec_logits, no_dec_col], dim=-1)  # (B, T, 10, 7)
    losses["next_decision"] = F.cross_entropy(
        dec_logits_full.view(-1, 7), batch["decision_labels"].view(-1).long()
    )

    mu = out["frame_mu"][:, :-1]      # (B, T-1, 10, 6)
    logvar = out["frame_logvar"][:, :-1]
    target = _frame_delta_labels(batch)    # (B, T-1, 10, 6)
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
    """Roll the prior forward ROLLOUT_STEPS from h_final/z_final; apply event +
    outcome heads on the rolled latents. Use last-observed labels as the
    multi-step target (cheap proxy; heads still see distribution shift)."""
    if ROLLOUT_STEPS == 0:
        return torch.tensor(0.0, device=out["event_logits"].device)
    steps = rollout_prior(model.rssm, out["h_final"], out["z_final"],
                          n_steps=ROLLOUT_STEPS, action_summary=None)
    total = 0.0
    # Gather last anchor's labels
    anchor_pos = batch["anchor_positions"]
    B, T = anchor_pos.shape
    idx = anchor_pos[:, -1:].unsqueeze(-1).expand(B, 1, batch["labels"].size(-1))
    last_labels = torch.gather(batch["labels"], 1, idx).squeeze(1).float()  # (B, NUM_EVENT_TYPES)
    last_outcome = _outcome_labels(batch)[:, -1]  # (B,)
    for (h, z, _mu, _lv), w in zip(steps, ROLLOUT_LOSS_WEIGHTS):
        ev_logits = model.head_event(z)
        oc_logits = model.head_outcome(z)
        total = total + w * (
            F.binary_cross_entropy_with_logits(ev_logits, last_labels) +
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
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true, y_score = [], []
    model.eval()
    for batch in loader:
        batch = _batch_to_device(batch, device)
        out = model(batch)
        T = out["n_anchors"]
        minute_idx = min(target_minute, T - 1)
        y_score.append(torch.sigmoid(out["outcome_logits"])[0, minute_idx].item())
        y_true.append(float(batch["outcome"][0].item()))
    model.train()
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def plan_b_train_loop(train_match_ids, val_match_ids, cold_match_ids,
                      epochs: int = 30, batch_size: int = 8, lr: float = 3e-4,
                      max_puuids: int = 20000,
                      log_every: int = 10,
                      checkpoint_tag: str = "plan_b_full"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    puuid_index = build_puuid_index(train_match_ids, max_puuids=max_puuids)

    # Exclude both holdouts from any feature aggregate queried during training.
    exclude = set(val_match_ids) | set(cold_match_ids)
    train_ds = MatchDataset(train_match_ids, puuid_index, exclude_match_ids=exclude)
    val_ds = MatchDataset(val_match_ids, puuid_index, exclude_match_ids=exclude)
    cold_ds = MatchDataset(cold_match_ids, puuid_index, exclude_match_ids=exclude) \
              if cold_match_ids else None

    device = _get_device()
    model = PlanBModel(max_puuids=max_puuids).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters()):,}")

    history = {"train_loss": [], "game_cold_auc15": [], "player_cold_auc15": []}
    best_cold_auc = -1.0
    patience_left = EARLY_STOP_PATIENCE
    best_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_tag}_best.pt")

    for ep in range(epochs):
        model.train()
        loader = DataLoader(train_ds, batch_size=batch_size,
                            collate_fn=collate_games, shuffle=True,
                            num_workers=4, persistent_workers=True,
                            prefetch_factor=2,
                            multiprocessing_context="forkserver")
        n_train_batches = len(loader)
        ep_start = time.time()
        ep_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(loader, start=1):
            batch = _batch_to_device(batch, device)
            out = model(batch)
            losses = _compute_losses(out, batch)
            aux = _rollout_aux_loss(model, out, batch)
            loss = _combined_loss(losses, aux)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
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

        print(f"epoch {ep+1}/{epochs} done  loss={ep_loss:.4f}  "
              f"game_cold_auc15={game_auc:.3f}  player_cold_auc15={cold_auc:.3f}")

        if cold_auc > best_cold_auc + 1e-4:
            best_cold_auc = cold_auc
            patience_left = EARLY_STOP_PATIENCE
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "max_puuids": max_puuids,
                        "history": history}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {ep+1} (no player-cold AUC@15 gain "
                      f"for {EARLY_STOP_PATIENCE} epochs).")
                break

    return history
