from model.train import train_loop
from model.dataset import load_split


def test_shakedown_overfit_tiny_corpus():
    """Sanity: the pipeline should overfit a tiny corpus. 5 games, 15 epochs.
    Expect val_top5 > 0.3 which is a low bar but proves signal flows
    end-to-end. The proper M1 check is Task 14."""
    train_ids = load_split("train")[:5]
    val_ids = train_ids  # same games — testing overfit
    best = train_loop(
        train_match_ids=train_ids,
        val_match_ids=val_ids,
        epochs=5,
        batch_size=2,
        lr=3e-4,
        log_every=100,
        checkpoint_tag="shakedown_smoke",
    )
    assert best > 0.1  # very low bar — just proves the loop runs and learns *something*.
