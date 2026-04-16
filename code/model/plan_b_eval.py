import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model.dataset import collate_games
from model.rollout import rollout_prior
from model.tokens import NUM_EVENT_TYPES


@torch.no_grad()
def outcome_auc_by_minute(model, ds, minutes=(5, 10, 15, 20, 25)):
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true = {m: [] for m in minutes}
    y_score = {m: [] for m in minutes}
    for batch in loader:
        out = model(batch)
        T = out["n_anchors"]
        truth = float(batch["outcome"][0].item())
        scores = torch.sigmoid(out["outcome_logits"])[0]
        for m in minutes:
            if m < T:
                y_true[m].append(truth)
                y_score[m].append(scores[m].item())
    model.train()
    result = {}
    for m in minutes:
        if len(set(y_true[m])) < 2 or not y_true[m]:
            result[m] = 0.5
        else:
            result[m] = float(roc_auc_score(y_true[m], y_score[m]))
    return result


@torch.no_grad()
def imagination_rollout_top5(model, ds, n_steps: int = 3):
    """For each held-out game, teacher-force up to the second-to-last anchor, roll
    the prior forward n_steps, and measure event top-5 accuracy averaged across
    step positions. Returns a list of length n_steps."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    per_step_hits = [[] for _ in range(n_steps)]

    for batch in loader:
        out = model(batch)
        T = out["n_anchors"]
        if T < 2 + n_steps:
            continue
        # Use posterior mean at anchor T - n_steps - 1 as seed z.
        post_mu = out["post_mu"][:, T - n_steps - 1]
        z_seed = post_mu  # mean, deterministic
        # Seed h from static encoder (approximation).
        h_seed = model.static_to_h(model.static_enc(batch["static"]))
        steps = rollout_prior(model.rssm, h_seed, z_seed, n_steps=n_steps,
                              action_summary=None)
        # Gather labels at anchor positions
        anchor_pos = batch["anchor_positions"]  # (1, T)
        for si, (_h, z, _mu, _lv) in enumerate(steps):
            target_t = T - n_steps + si
            # Get label at this anchor position from the full labels tensor
            pos = anchor_pos[0, target_t].item()
            target = batch["labels"][0, pos]  # (NUM_EVENT_TYPES,) multi-hot
            logits = model.head_event(z)[0]   # (NUM_EVENT_TYPES,)
            top5 = torch.topk(logits, k=5).indices.tolist()
            argmax_class = int(torch.argmax(target).item())
            per_step_hits[si].append(1.0 if argmax_class in top5 else 0.0)

    model.train()
    return [float(sum(hits) / max(1, len(hits))) for hits in per_step_hits]


@torch.no_grad()
def frozen_minute_0_auc(model, ds, target_minute: int = 15) -> float:
    """Feed only static + player streams; zero out the dynamic sequence. Predict
    outcome at `target_minute`. Target AUC <= 0.55 (chance-like)."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true, y_score = [], []
    for batch in loader:
        frozen_batch = {k: v for k, v in batch.items()}
        frozen_batch["tokens"] = torch.zeros_like(batch["tokens"])
        frozen_batch["event_window_embeddings_raw"] = torch.zeros_like(
            batch["event_window_embeddings_raw"])
        frozen_batch["window_mask"] = torch.zeros_like(batch["window_mask"])
        frozen_batch["frame_features"] = torch.zeros_like(batch["frame_features"])
        out = model(frozen_batch)
        T = out["n_anchors"]
        if target_minute < T:
            y_true.append(float(batch["outcome"][0].item()))
            y_score.append(torch.sigmoid(out["outcome_logits"])[0, target_minute].item())
    model.train()
    if len(set(y_true)) < 2 or not y_true:
        return 0.5
    return float(roc_auc_score(y_true, y_score))
