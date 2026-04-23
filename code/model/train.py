"""Training loop for the baseline classifier."""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dataset import (
    MatchDataset, collate_games, load_split, build_puuid_index,
)
from model.baseline import CausalTransformerBaseline
from model.tokens import NUM_EVENT_TYPES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'model_checkpoints')


def masked_bce_loss(logits, labels, mask):
    """Binary cross-entropy averaged over masked positions only."""
    per_pos = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction='none'
    ).mean(dim=-1)  # (B, L)
    denom = mask.sum().clamp(min=1.0)
    return (per_pos * mask).sum() / denom


def top5_accuracy(logits, labels, mask):
    """Fraction of masked positions where top-5 predicted classes cover the
    majority of the ground-truth positive set. Specifically: position i
    counts as a hit if top-5 predictions include the single most-common
    ground-truth class at that position. If no positives at that position,
    skip."""
    B, L, C = logits.shape
    topk = logits.topk(k=min(5, C), dim=-1).indices  # (B, L, 5)
    top_true = labels.argmax(dim=-1)  # (B, L) — dominant class per position
    pos_mask = (labels.sum(dim=-1) > 0) & mask.bool()
    hits = (topk == top_true.unsqueeze(-1)).any(dim=-1) & pos_mask
    denom = pos_mask.sum().clamp(min=1)
    return hits.sum().float() / denom.float()


def train_loop(
    train_match_ids,
    val_match_ids,
    epochs=20,
    batch_size=8,
    lr=3e-4,
    device=None,
    max_puuids=20000,
    log_every=10,
    checkpoint_tag="baseline",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train: {len(train_match_ids)}  Val: {len(val_match_ids)}")

    puuid_index = build_puuid_index(train_match_ids, max_puuids=max_puuids)
    print(f"Built puuid index with {len(puuid_index)} known puuids")

    train_ds = MatchDataset(train_match_ids, puuid_index=puuid_index)
    val_ds = MatchDataset(val_match_ids, puuid_index=puuid_index)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_games, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_games, num_workers=0,
    )

    model = CausalTransformerBaseline(max_puuids=max_puuids).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_top5 = 0.0

    for epoch in range(epochs):
        model.train()
        ep_start = time.time()
        tot_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            loss = masked_bce_loss(logits, batch["labels"], batch["label_mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            n_batches += 1
            if step % log_every == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_loss = tot_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_top5 = 0.0
            v_batches = 0
            for batch in val_dl:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = model(batch)
                val_loss += masked_bce_loss(logits, batch["labels"], batch["label_mask"]).item()
                val_top5 += top5_accuracy(logits, batch["labels"], batch["label_mask"]).item()
                v_batches += 1
            val_loss /= max(v_batches, 1)
            val_top5 /= max(v_batches, 1)

        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
              f"val_top5={val_top5:.4f}  ({time.time()-ep_start:.1f}s)")

        if val_top5 > best_val_top5:
            best_val_top5 = val_top5
            ckpt = os.path.join(CHECKPOINT_DIR, f"{checkpoint_tag}_best.pt")
            torch.save({
                "model": model.state_dict(),
                "puuid_index": puuid_index,
                "val_top5": val_top5,
                "epoch": epoch,
            }, ckpt)
            print(f"  saved best checkpoint -> {ckpt}")

    return best_val_top5
