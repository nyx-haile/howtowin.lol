import model.cli as cli
import model.plan_b_train as plan_b_train_module


def test_plan_b_train_cli_forwards_runtime_knobs(monkeypatch):
    captured = {}

    def fake_load_split(name):
        if name == "train":
            return ["train-1", "train-2"]
        if name == "holdout":
            return ["val-1"]
        if name == "cold":
            return ["cold-1"]
        raise AssertionError(f"unexpected split {name}")

    def fake_plan_b_train_loop(train, val, cold, **kwargs):
        captured["train"] = train
        captured["val"] = val
        captured["cold"] = cold
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "load_split", fake_load_split)
    monkeypatch.setattr(plan_b_train_module, "plan_b_train_loop", fake_plan_b_train_loop)

    args = cli.build_parser().parse_args(
        [
            "plan-b-train",
            "--epochs", "2",
            "--batch-size", "3",
            "--num-workers", "12",
            "--prefetch-factor", "4",
            "--no-persistent-workers",
            "--loader-order", "out-of-order",
            "--train-cache-size", "0",
        ]
    )

    cli.cmd_plan_b_train(args)

    assert captured["train"] == ["train-1", "train-2"]
    assert captured["val"] == ["val-1"]
    assert captured["cold"] == ["cold-1"]
    assert captured["kwargs"]["epochs"] == 2
    assert captured["kwargs"]["batch_size"] == 3
    assert captured["kwargs"]["num_workers"] == 12
    assert captured["kwargs"]["prefetch_factor"] == 4
    assert captured["kwargs"]["persistent_workers"] is False
    assert captured["kwargs"]["loader_order"] == "out-of-order"
    assert captured["kwargs"]["train_cache_size"] == 0
