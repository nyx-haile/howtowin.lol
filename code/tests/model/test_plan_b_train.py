from model.plan_b_train import plan_b_train_loop


def test_overfit_tiny_corpus_drives_loss_down():
    """Smoke test: 5 games x 3 epochs must drive train loss below its start."""
    from model.dataset import load_split
    train_ids = load_split("train")[:5]
    val_ids = load_split("holdout")[:2]
    cold_ids = load_split("cold")[:2] if load_split("cold") else []

    history = plan_b_train_loop(
        train_ids, val_ids, cold_ids,
        epochs=3, batch_size=1, lr=1e-3,
        max_puuids=50,
        checkpoint_tag="plan_b_smoke",
    )
    first, last = history["train_loss"][0], history["train_loss"][-1]
    assert last < first, f"smoke test: train loss did not decrease ({first} -> {last})"
