import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model.dataset import collate_games
from model.rollout import rollout_prior


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
    """Teacher-force up to an anchor, roll forward in latent space, and measure
    coarse event-type top-5 accuracy for future anchors."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    per_step_hits = [[] for _ in range(n_steps)]

    for batch in loader:
        out = model(batch)
        T = out["n_anchors"]
        if T < 2 + n_steps:
            continue
        seed_t = T - n_steps - 1
        z_seed = out["post_mu"][:, seed_t]
        h_seed = out["h"][:, seed_t]
        steps = rollout_prior(model.rssm, h_seed, z_seed, n_steps=n_steps, action_summary=None)
        for si, (_h, z, _mu, _lv) in enumerate(steps):
            target_t = T - n_steps + si
            target = int(batch["next_event_type_labels"][0, target_t].item())
            if target <= 0:
                continue
            type_logits = model.head_event(z)["type_logits"][0, 1:]
            top5 = torch.topk(type_logits, k=min(5, type_logits.numel())).indices.tolist()
            per_step_hits[si].append(1.0 if (target - 1) in top5 else 0.0)

    model.train()
    return [float(sum(hits) / max(1, len(hits))) for hits in per_step_hits]


@torch.no_grad()
def frozen_minute_0_auc(model, ds, target_minute: int = 15) -> float:
    """Feed only static + player streams; zero out the dynamic sequence."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true, y_score = [], []
    for batch in loader:
        frozen_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        frozen_batch["tokens"] = torch.zeros_like(batch["tokens"])
        frozen_batch["token_actors"] = torch.zeros_like(batch["token_actors"])
        frozen_batch["token_targets"] = torch.zeros_like(batch["token_targets"])
        frozen_batch["token_item_ids"] = torch.zeros_like(batch["token_item_ids"])
        frozen_batch["token_skill_slots"] = torch.zeros_like(batch["token_skill_slots"])
        frozen_batch["token_monster_types"] = torch.zeros_like(batch["token_monster_types"])
        frozen_batch["token_monster_subtypes"] = torch.zeros_like(batch["token_monster_subtypes"])
        frozen_batch["token_building_types"] = torch.zeros_like(batch["token_building_types"])
        frozen_batch["token_lane_types"] = torch.zeros_like(batch["token_lane_types"])
        frozen_batch["token_tower_types"] = torch.zeros_like(batch["token_tower_types"])
        frozen_batch["token_ward_types"] = torch.zeros_like(batch["token_ward_types"])
        frozen_batch["event_window_embeddings_raw"] = torch.zeros_like(batch["event_window_embeddings_raw"])
        frozen_batch["window_mask"] = torch.zeros_like(batch["window_mask"])
        frozen_batch["frame_features"] = torch.zeros_like(batch["frame_features"])
        frozen_batch["anchor_macro_features"] = torch.zeros_like(batch["anchor_macro_features"])
        out = model(frozen_batch)
        T = out["n_anchors"]
        if target_minute < T:
            y_true.append(float(batch["outcome"][0].item()))
            y_score.append(torch.sigmoid(out["outcome_logits"])[0, target_minute].item())
    model.train()
    if len(set(y_true)) < 2 or not y_true:
        return 0.5
    return float(roc_auc_score(y_true, y_score))
