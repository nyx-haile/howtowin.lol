"""Command-line entry points for Plan A and Plan B.

Usage:
  python -m model.cli shakedown              # Plan A M1 overfit check
  python -m model.cli train [--epochs N]     # Plan A full train
  python -m model.cli eval                   # Plan A held-out eval
  python -m model.cli cold-build             # Build player-cold holdout file
  python -m model.cli plan-b-shakedown       # Plan B M1 overfit check
  python -m model.cli plan-b-train [--epochs N]  # Plan B full train
  python -m model.cli plan-b-eval            # Plan B held-out eval
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


def cmd_cold_build(args):
    from model.cold_holdout import save_player_cold_holdout
    puuids, match_ids = save_player_cold_holdout()
    print(f"cold puuids: {len(puuids)}  cold matches: {len(match_ids)}")


def cmd_plan_b_shakedown(args):
    from model.plan_b_train import plan_b_train_loop
    train = load_split("train")[:50]
    val = load_split("holdout")[:8]
    cold = load_split("cold")[:8]
    hist = plan_b_train_loop(train, val, cold,
                             epochs=30, batch_size=2, lr=1e-3,
                             max_puuids=500,
                             log_every=args.log_every,
                             checkpoint_tag="plan_b_shakedown")
    best_cold = max(hist["player_cold_auc15"])
    print(f"best player_cold_auc15 = {best_cold:.3f}")
    assert best_cold >= 0.55, "shakedown failed: player-cold AUC below floor"


def cmd_plan_b_train(args):
    from model.plan_b_train import plan_b_train_loop
    train = load_split("train")
    val = load_split("holdout")
    cold = load_split("cold")
    plan_b_train_loop(train, val, cold,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, max_puuids=args.max_puuids,
                      log_every=args.log_every,
                      checkpoint_tag="plan_b_full")


def cmd_plan_b_eval(args):
    from model.plan_b_model import PlanBModel
    from model.plan_b_eval import (
        outcome_auc_by_minute, imagination_rollout_top5, frozen_minute_0_auc,
    )
    ckpt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "model_checkpoints",
        "plan_b_full_best.pt")
    ckpt = torch.load(ckpt_path, weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)

    val_ids = load_split("holdout")
    cold_ids = load_split("cold")
    idx = build_puuid_index(load_split("train"), max_puuids=max_puuids)

    exclude = set(val_ids) | set(cold_ids)
    game_ds = MatchDataset(val_ids, idx, exclude_match_ids=exclude)
    cold_ds = MatchDataset(cold_ids, idx, exclude_match_ids=exclude)

    model = PlanBModel(max_puuids=max_puuids)
    model.load_state_dict(ckpt["state_dict"])

    print("=== GAME-COLD HOLDOUT ===")
    for m, v in outcome_auc_by_minute(model, game_ds).items():
        print(f"  minute {m:>2}: AUC={v:.3f}")

    print("=== PLAYER-COLD HOLDOUT ===")
    for m, v in outcome_auc_by_minute(model, cold_ds).items():
        print(f"  minute {m:>2}: AUC={v:.3f}")

    print("=== IMAGINATION ROLLOUT (game-cold) ===")
    for si, top5 in enumerate(imagination_rollout_top5(model, game_ds, n_steps=3), start=1):
        print(f"  step {si}: event top-5 = {top5:.3f}")

    print("=== LEAK PROBE (frozen-minute-0) ===")
    auc = frozen_minute_0_auc(model, game_ds, target_minute=15)
    print(f"  AUC @15 = {auc:.3f}  (target: <= 0.55)")


def cmd_eval():
    """M2 acceptance: top-5 >= 0.95 (stretch target) on held-out."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "baseline_full_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    puuid_index = ckpt["puuid_index"]
    max_puuids = ckpt["model"]["player_enc.residual.weight"].shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CausalTransformerBaseline(max_puuids=max_puuids).to(device)
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

    sub.add_parser("cold-build")
    p_pb_sd = sub.add_parser("plan-b-shakedown")
    p_pb_sd.add_argument("--log-every", type=int, default=10, dest="log_every")
    p_pb_tr = sub.add_parser("plan-b-train")
    p_pb_tr.add_argument("--epochs", type=int, default=30)
    p_pb_tr.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p_pb_tr.add_argument("--lr", type=float, default=3e-4)
    p_pb_tr.add_argument("--max-puuids", type=int, default=20000, dest="max_puuids")
    p_pb_tr.add_argument("--log-every", type=int, default=10, dest="log_every")
    sub.add_parser("plan-b-eval")

    args = parser.parse_args()

    if args.cmd == "shakedown":
        ok = cmd_shakedown()
        sys.exit(0 if ok else 1)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval()
    elif args.cmd == "cold-build":
        cmd_cold_build(args)
    elif args.cmd == "plan-b-shakedown":
        cmd_plan_b_shakedown(args)
    elif args.cmd == "plan-b-train":
        cmd_plan_b_train(args)
    elif args.cmd == "plan-b-eval":
        cmd_plan_b_eval(args)
