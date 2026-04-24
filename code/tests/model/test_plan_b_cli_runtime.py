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


def test_plan_b_train_cli_forwards_preflight_and_cache_knobs(monkeypatch):
    captured = {}

    def fake_load_split(name):
        return []

    def fake_plan_b_train_loop(train, val, cold, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "load_split", fake_load_split)
    monkeypatch.setattr(plan_b_train_module, "plan_b_train_loop", fake_plan_b_train_loop)

    args = cli.build_parser().parse_args(
        [
            "plan-b-train",
            "--preflight-only",
            "--no-preflight-strict",
            "--preflight-artifact", "/tmp/prefl.json",
            "--preflight-max-earlystop-hours", "24.0",
            "--preflight-max-epoch-minutes", "180.0",
            "--materialized-cache-dir", "/tmp/cache",
            "--materialized-cache-mode", "readwrite",
            "--materialized-cache-version", "v1-remote",
            "--materialized-cache-warmup",
        ]
    )
    cli.cmd_plan_b_train(args)

    kw = captured["kwargs"]
    assert kw["preflight_only"] is True
    assert kw["preflight_strict"] is False
    assert kw["preflight_artifact_path"] == "/tmp/prefl.json"
    assert kw["preflight_max_earlystop_hours"] == 24.0
    assert kw["preflight_max_epoch_minutes"] == 180.0
    assert kw["materialized_cache_dir"] == "/tmp/cache"
    assert kw["materialized_cache_mode"] == "readwrite"
    assert kw["materialized_cache_version"] == "v1-remote"
    assert kw["materialized_cache_warmup"] is True
