import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model.dataset import collate_games


PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT = (
    "PlanBModel.forward samples RSSM latents during eval; these helpers keep "
    "PlanB model forwards at DataLoader batch_size=1 until a deterministic "
    "or RNG-replay contract exists."
)


def _record_counter(counters: dict | None, name: str, amount: int = 1) -> None:
    if counters is None:
        return
    counters[name] = int(counters.get(name, 0)) + amount


def _record_model_forward(counters: dict | None, batch: dict) -> None:
    """Record the explicit no-batched-PlanB-forward evaluation contract."""
    if counters is None:
        return
    _record_counter(counters, "model_forward_calls")
    batch_size = 1
    tokens = batch.get("tokens")
    if torch.is_tensor(tokens):
        batch_size = int(tokens.size(0))
    counters["max_model_forward_batch_size"] = max(
        int(counters.get("max_model_forward_batch_size", 0)),
        batch_size,
    )
    counters.setdefault("stochastic_planb_forward_batching", "disabled_batch_size_1")
    counters.setdefault("stochastic_planb_forward_caveat", PLANB_STOCHASTIC_EVAL_BATCHING_CAVEAT)


@torch.no_grad()
def outcome_auc_by_minute(model, ds, minutes=(5, 10, 15, 20, 25), *, counters: dict | None = None):
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true = {m: [] for m in minutes}
    y_score = {m: [] for m in minutes}
    for batch in loader:
        out = model(batch)
        _record_model_forward(counters, batch)
        n_anchors = int(batch["anchor_mask"][0].sum().item())
        truth = float(batch["outcome"][0].item())
        scores = torch.sigmoid(out["outcome_logits"])[0]
        for m in minutes:
            if m < n_anchors:
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
def imagination_rollout_top5(model, ds, n_steps: int = 3, *, counters: dict | None = None):
    """Teacher-force to one game at a time, then roll forward with true future event windows."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    per_step_hits = [[] for _ in range(n_steps)]

    for batch in loader:
        out = model(batch)
        _record_model_forward(counters, batch)
        n_anchors = int(batch["anchor_mask"][0].sum().item())
        if n_anchors < 2 + n_steps:
            continue

        seed_t = n_anchors - n_steps - 1
        _record_counter(counters, "rollout_prior_single_calls")
        steps = model.rollout_prior_single(
            h0=out["h"][0, seed_t],
            z0=out["z"][0, seed_t],
            static_tokens=out["static_tokens"][0],
            token_embeddings=out["token_embeddings"][0],
            event_window_positions=batch["event_window_positions"],
            event_window_offsets=batch["event_window_offsets"][0],
            event_window_counts=batch["event_window_counts"][0],
            anchor_mask=batch["anchor_mask"][0],
            start_anchor=seed_t,
            n_steps=n_steps,
        )
        for si, step in enumerate(steps):
            target_t = seed_t + si + 1
            target = int(batch["next_event_type_labels"][0, target_t].item())
            if target <= 0:
                continue
            decoded = model.decode_anchor_repr(step["anchor_repr"])
            type_logits = decoded["event_factors"]["type_logits"][0, 1:]
            top5 = torch.topk(type_logits, k=min(5, type_logits.numel())).indices.tolist()
            per_step_hits[si].append(1.0 if (target - 1) in top5 else 0.0)

    model.train()
    return [float(sum(hits) / max(1, len(hits))) for hits in per_step_hits]


@torch.no_grad()
def frozen_minute_0_auc(model, ds, target_minute: int = 15, *, counters: dict | None = None) -> float:
    """Feed only static + player streams; zero out the dynamic sequence."""
    model.eval()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_games, shuffle=False)
    y_true, y_score = [], []
    for batch in loader:
        frozen_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        frozen_batch["tokens"] = torch.zeros_like(batch["tokens"])
        frozen_batch["token_actors"] = torch.zeros_like(batch["token_actors"])
        frozen_batch["token_targets"] = torch.zeros_like(batch["token_targets"])
        frozen_batch["token_timestamps"] = torch.zeros_like(batch["token_timestamps"])
        frozen_batch["token_item_ids"] = torch.zeros_like(batch["token_item_ids"])
        frozen_batch["token_skill_slots"] = torch.zeros_like(batch["token_skill_slots"])
        frozen_batch["token_monster_types"] = torch.zeros_like(batch["token_monster_types"])
        frozen_batch["token_monster_subtypes"] = torch.zeros_like(batch["token_monster_subtypes"])
        frozen_batch["token_building_types"] = torch.zeros_like(batch["token_building_types"])
        frozen_batch["token_lane_types"] = torch.zeros_like(batch["token_lane_types"])
        frozen_batch["token_tower_types"] = torch.zeros_like(batch["token_tower_types"])
        frozen_batch["token_ward_types"] = torch.zeros_like(batch["token_ward_types"])
        frozen_batch["event_window_positions"] = torch.zeros(0, dtype=torch.long)
        frozen_batch["event_window_offsets"] = torch.zeros_like(batch["event_window_offsets"])
        frozen_batch["event_window_counts"] = torch.zeros_like(batch["event_window_counts"])
        frozen_batch["frame_features"] = torch.zeros_like(batch["frame_features"])
        frozen_batch["anchor_macro_features"] = torch.zeros_like(batch["anchor_macro_features"])
        out = model(frozen_batch)
        _record_model_forward(counters, frozen_batch)
        n_anchors = int(batch["anchor_mask"][0].sum().item())
        if target_minute < n_anchors:
            y_true.append(float(batch["outcome"][0].item()))
            y_score.append(torch.sigmoid(out["outcome_logits"])[0, target_minute].item())
    model.train()
    if len(set(y_true)) < 2 or not y_true:
        return 0.5
    return float(roc_auc_score(y_true, y_score))
