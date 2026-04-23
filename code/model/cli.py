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
                      num_workers=args.num_workers,
                      prefetch_factor=args.prefetch_factor,
                      persistent_workers=args.persistent_workers,
                      loader_order=args.loader_order,
                      train_cache_size=args.train_cache_size,
                      checkpoint_tag="plan_b_full",
                      compile_model=args.compile_model)


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


def _checkpoint_sha_short(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def cmd_retrieval_build(args):
    """Build the M4 retrieval index (model + optional baselines)."""
    import torch as _t
    from model.plan_b_model import PlanBModel
    from model.retrieval import (
        build_index, save_index, DEFAULT_INDEX_PATH, INDEX_DIR,
    )

    ckpt_path = os.path.join(CHECKPOINT_DIR, "plan_b_full_best.pt")
    ckpt = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)
    sha = _checkpoint_sha_short(ckpt_path)

    train = load_split("train")
    val = load_split("holdout")
    cold = load_split("cold")
    exclude = set(val) | set(cold)
    puuid_index = build_puuid_index(train, max_puuids=max_puuids)

    device = "cuda" if _t.cuda.is_available() else "cpu"
    model = PlanBModel(max_puuids=max_puuids).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print(f"[retrieval-build] device={device} train={len(train)} "
          f"excluded={len(exclude)} ckpt_sha={sha}")
    bundle = build_index(
        model=model, train_match_ids=train,
        exclude_match_ids=exclude, puuid_index=puuid_index,
        device=device, checkpoint_sha=sha,
    )
    save_index(bundle, DEFAULT_INDEX_PATH)
    print(f"[retrieval-build] saved {DEFAULT_INDEX_PATH} "
          f"corpus_rows={bundle.corpus_white.shape[0]}")

    if args.baselines:
        from model.baselines.static_only_index import build_static_only_index
        from model.baselines.frame_features_index import build_frame_features_index

        so_path = os.path.join(INDEX_DIR, "plan_b_static_only_index.pt")
        ff_path = os.path.join(INDEX_DIR, "plan_b_frame_features_index.pt")

        print("[retrieval-build] static-only baseline...")
        so_bundle = build_static_only_index(
            model=model, train_match_ids=train,
            exclude_match_ids=exclude, puuid_index=puuid_index,
            device=device,
        )
        save_index(so_bundle, so_path)
        print(f"[retrieval-build] saved {so_path} "
              f"corpus_rows={so_bundle.corpus_white.shape[0]}")

        print("[retrieval-build] frame-features baseline...")
        ff_bundle = build_frame_features_index(
            train_match_ids=train, exclude_match_ids=exclude,
        )
        save_index(ff_bundle, ff_path)
        print(f"[retrieval-build] saved {ff_path} "
              f"corpus_rows={ff_bundle.corpus_white.shape[0]}")


def cmd_retrieval_eval(args):
    """Run the M4 eval and write a markdown report."""
    import torch as _t
    from datetime import date

    from model.plan_b_model import PlanBModel
    from model.retrieval import (
        load_index, DEFAULT_INDEX_PATH, INDEX_DIR,
        K_SWEEP, HEADLINE_K, HEADLINE_GATE_BITS, MID_GAME_MINUTES,
    )
    from model.m4_eval import run_m4_eval

    ckpt_path = os.path.join(CHECKPOINT_DIR, "plan_b_full_best.pt")
    ckpt = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)

    train = load_split("train")
    val = load_split("holdout")
    cold = load_split("cold")
    exclude = set(val) | set(cold)
    puuid_index = build_puuid_index(train, max_puuids=max_puuids)

    device = "cuda" if _t.cuda.is_available() else "cpu"
    model = PlanBModel(max_puuids=max_puuids).to(device)
    model.load_state_dict(ckpt["state_dict"])

    model_bundle = load_index(DEFAULT_INDEX_PATH)
    so_bundle = ff_bundle = None
    if args.baselines:
        so_bundle = load_index(os.path.join(INDEX_DIR, "plan_b_static_only_index.pt"))
        ff_bundle = load_index(os.path.join(INDEX_DIR, "plan_b_frame_features_index.pt"))

    results = {}
    for label, ids in [("game_cold", val), ("player_cold", cold)]:
        print(f"\n=== {label.upper()} ({len(ids)} games) ===")
        r = run_m4_eval(
            model=model, model_bundle=model_bundle,
            holdout_match_ids=ids, holdout_label=label,
            puuid_index=puuid_index, exclude_match_ids=exclude,
            k_sweep=K_SWEEP, headline_k=HEADLINE_K,
            headline_minutes=MID_GAME_MINUTES,
            device=device, query_batch_size=128,
            run_baselines=args.baselines,
            static_only_bundle=so_bundle,
            frame_features_bundle=ff_bundle,
        )
        results[label] = r
        _print_eval_result(r)

    passed = (
        results["game_cold"]["model"]["k_sweep"][HEADLINE_K] >= HEADLINE_GATE_BITS
        and results["player_cold"]["model"]["k_sweep"][HEADLINE_K] >= HEADLINE_GATE_BITS
    )
    print(f"\nM4 GATE: {'PASS' if passed else 'FAIL'} "
          f"(headline_k={HEADLINE_K}, threshold={HEADLINE_GATE_BITS} bits)")

    report_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "docs",
    )
    report_path = os.path.join(
        report_dir, f"m4_retrieval_eval_report_{date.today().isoformat()}.md",
    )
    _write_report(report_path, results, passed)
    print(f"[retrieval-eval] wrote report to {report_path}")


def _print_eval_result(r: dict) -> None:
    n_q = r["n_queries"]
    print(f"  n_queries={n_q}")
    print("  Source            " + "  ".join(f"k={k:>3}" for k in r["model"]["k_sweep"]))
    for src in ("model", "static_only", "frame_features", "random"):
        if src in r:
            row = r[src]["k_sweep"]
            cells = "  ".join(f"{row[k]:.3f}" for k in row)
            print(f"  {src:<16}  {cells}")
    print("  Per-minute entropy at headline k (model only):")
    for m, v in sorted(r["model"]["per_minute_at_headline_k"].items()):
        print(f"    min {m:>2}: {v:.3f}" if v == v else f"    min {m:>2}:   nan")


def _write_report(path: str, results: dict, passed: bool) -> None:
    from datetime import date
    lines = [
        "# M4 Retrieval-Check Eval Report",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Gate:** mean cohort outcome entropy ≥ 0.7 bits at k=64 on both holdouts.",
        f"**Result:** **{'PASS' if passed else 'FAIL'}**",
        "",
    ]
    for label in ("game_cold", "player_cold"):
        r = results[label]
        lines.append(f"## {label} (n_queries={r['n_queries']})")
        ks = list(r["model"]["k_sweep"].keys())
        header = "| Source | " + " | ".join(f"k={k}" for k in ks) + " |"
        sep = "|" + "---|" * (len(ks) + 1)
        lines += ["", header, sep]
        for src in ("model", "static_only", "frame_features", "random"):
            if src in r:
                row = r[src]["k_sweep"]
                cells = " | ".join(f"{row[k]:.3f}" for k in ks)
                lines.append(f"| {src} | {cells} |")
        lines.append("")
        lines.append("### Per-minute entropy at headline k (model only)")
        lines.append("")
        lines.append("| Minute | Entropy (bits) |")
        lines.append("|---:|---:|")
        for m, v in sorted(r["model"]["per_minute_at_headline_k"].items()):
            cell = f"{v:.3f}" if v == v else "nan"
            lines.append(f"| {m} | {cell} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def cmd_lesson(args):
    """Produce mistake + strength anchor candidates for one game."""
    import json
    from model.lesson import generate_lesson, lesson_to_dict

    res = generate_lesson(
        args.match_id, team=args.team, k=args.k,
        index_path=args.index_path, ckpt_path=args.ckpt_path,
    )
    print(json.dumps(lesson_to_dict(res), indent=2))


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


def build_parser() -> argparse.ArgumentParser:
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
    p_pb_tr.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p_pb_tr.add_argument("--prefetch-factor", type=int, default=None, dest="prefetch_factor")
    p_pb_tr.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="persistent_workers",
    )
    p_pb_tr.add_argument(
        "--loader-order",
        choices=("auto", "ordered", "out-of-order"),
        default="auto",
        dest="loader_order",
    )
    p_pb_tr.add_argument("--train-cache-size", type=int, default=None, dest="train_cache_size")
    p_pb_tr.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="compile_model",
    )
    sub.add_parser("plan-b-eval")

    p_rb = sub.add_parser("retrieval-build")
    p_rb.add_argument("--baselines", action="store_true",
                      help="also build static-only and frame-features indexes")

    p_re = sub.add_parser("retrieval-eval")
    p_re.add_argument("--baselines", action="store_true",
                      help="also evaluate baselines (requires --baselines on build)")

    p_ls = sub.add_parser("lesson", help="Generate lesson-anchor candidates for one game")
    p_ls.add_argument("--match-id", required=True, dest="match_id")
    p_ls.add_argument("--team", choices=("blue", "red"), default="blue")
    p_ls.add_argument("--k", type=int, default=64)
    p_ls.add_argument("--index-path", default=None, dest="index_path")
    p_ls.add_argument("--ckpt-path", default=None, dest="ckpt_path")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

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
    elif args.cmd == "retrieval-build":
        cmd_retrieval_build(args)
    elif args.cmd == "retrieval-eval":
        cmd_retrieval_eval(args)
    elif args.cmd == "lesson":
        cmd_lesson(args)


if __name__ == "__main__":
    main()
