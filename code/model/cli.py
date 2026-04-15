"""Command-line entry points for Plan A.

Usage:
  python -m model.cli shakedown          # M1 overfit-tiny-corpus check
  python -m model.cli train [--epochs N] # full train on train split
  python -m model.cli eval               # M2 held-out evaluation using best ckpt
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import (
    MatchDataset, collate_games, load_split, build_puuid_index,
)
from model.train import train_loop, CHECKPOINT_DIR, masked_bce_loss
from model.baseline import CausalTransformerBaseline
from model.eval import top5_by_minute
from torch.utils.data import DataLoader


def cmd_shakedown():
    """M1 acceptance: overfit a 50-game corpus near-zero loss."""
    train_ids = load_split("train")[:50]
    best = train_loop(
        train_match_ids=train_ids,
        val_match_ids=train_ids,
        epochs=30,
        batch_size=4,
        lr=3e-4,
        log_every=20,
        checkpoint_tag="shakedown",
    )
    print(f"\nSHAKEDOWN best train-set top5: {best:.4f}")
    print("M1 acceptance: best >= 0.9 (near-overfit on 50 games)")
    return best >= 0.9


def cmd_train(args):
    train_ids = load_split("train")
    val_ids = load_split("holdout")
    best = train_loop(
        train_match_ids=train_ids,
        val_match_ids=val_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=50,
        checkpoint_tag="baseline_full",
    )
    print(f"\nFULL TRAIN best val top5: {best:.4f}")


def cmd_eval():
    """M2 acceptance: top-5 >= 0.95 (stretch target) on held-out."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "baseline_full_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    puuid_index = ckpt["puuid_index"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CausalTransformerBaseline(max_puuids=len(puuid_index) + 1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ids = load_split("holdout")
    val_ds = MatchDataset(val_ids, puuid_index=puuid_index)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_games)

    all_results = {"overall": 0.0, "by_minute": {}}
    n_batches = 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            r = top5_by_minute(
                logits, batch["labels"], batch["label_mask"], batch["token_timestamps"],
            )
            all_results["overall"] += r["overall"]
            for m, acc in r["by_minute"].items():
                all_results["by_minute"].setdefault(m, []).append(acc)
            n_batches += 1

    overall = all_results["overall"] / max(n_batches, 1)
    print(f"\nHELD-OUT top-5 (overall): {overall:.4f}")
    print("By minute:")
    for m in sorted(all_results["by_minute"].keys()):
        accs = all_results["by_minute"][m]
        print(f"  min {m:3d}: {sum(accs)/len(accs):.4f} (n={len(accs)})")
    print(f"\nM2 stretch target: overall >= 0.95")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shakedown")
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--epochs", type=int, default=30)
    p_tr.add_argument("--batch_size", type=int, default=8)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    sub.add_parser("eval")
    args = parser.parse_args()

    if args.cmd == "shakedown":
        ok = cmd_shakedown()
        sys.exit(0 if ok else 1)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval()
