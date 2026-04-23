#!/usr/bin/env python3
"""B4.1 verification: 2000-match subset, 2 epochs. Check no monotonic step-time growth in epoch 2."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

if __name__ == "__main__":
    from model.dataset import load_split
    from model.plan_b_train import plan_b_train_loop

    train_all = load_split("train")
    val = load_split("holdout")
    cold = load_split("cold")

    # Deterministic 2000-match subset
    train_subset = train_all[:2000]
    print(f"Verification subset: {len(train_subset)} train  {len(val)} val  {len(cold)} cold", flush=True)

    plan_b_train_loop(
        train_subset, val, cold,
        epochs=2,
        batch_size=4,
        log_every=5,
        num_workers=12,
        checkpoint_tag="plan_b_verify",
        run_preflight=False,
        compile_model=False,
    )
