import pytest
import torch

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
            "--max-steps", "2",
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
    assert captured["kwargs"]["max_steps"] == 2


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


def test_plan_b_train_cli_default_preflight_artifact_uses_output_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HOWL_OUTPUT_ROOT", str(tmp_path))

    args = cli.build_parser().parse_args(["plan-b-train"])

    assert args.preflight_artifact == str(tmp_path / "artifacts" / "runtime_preflight.json")


def test_retrieval_eval_parser_exposes_runtime_knobs():
    args = cli.build_parser().parse_args(
        [
            "retrieval-eval",
            "--device", "cpu",
            "--query-batch-size", "7",
        ]
    )

    assert args.device == "cpu"
    assert args.query_batch_size == 7

    defaults = cli.build_parser().parse_args(["retrieval-eval"])
    assert defaults.device is None
    assert defaults.query_batch_size == 128


def test_retrieval_eval_forwards_device_and_query_batch_size(monkeypatch, capsys):
    import model.m4_eval as m4_eval_module
    import model.plan_b_model as plan_b_model_module
    import model.retrieval as retrieval_module

    captured = {}

    class FakeModel:
        def __init__(self, max_puuids):
            captured["max_puuids"] = max_puuids

        def to(self, device):
            captured["model_device"] = device
            return self

        def load_state_dict(self, state_dict):
            captured["state_dict"] = state_dict

    def fake_load_split(name):
        return {
            "train": ["train-1"],
            "holdout": ["game-cold-1"],
            "cold": ["player-cold-1"],
        }[name]

    def fake_run_m4_eval(**kwargs):
        captured.setdefault("runs", []).append(kwargs)
        return {
            "n_queries": 1,
            "model": {
                "k_sweep": {64: 0.8},
                "per_minute_at_headline_k": {10: 0.8},
            },
        }

    monkeypatch.setattr(torch, "load", lambda *a, **kw: {
        "max_puuids": 123,
        "state_dict": {"weight": "fake"},
    })
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(plan_b_model_module, "PlanBModel", FakeModel)
    monkeypatch.setattr(retrieval_module, "load_index", lambda path: {
        "path": path,
    })
    monkeypatch.setattr(m4_eval_module, "run_m4_eval", fake_run_m4_eval)
    monkeypatch.setattr(cli, "load_split", fake_load_split)
    monkeypatch.setattr(cli, "build_puuid_index", lambda ids, max_puuids: {
        "ids": ids,
        "max_puuids": max_puuids,
    })
    monkeypatch.setattr(cli, "_print_eval_result", lambda r: None)
    monkeypatch.setattr(cli, "_write_report", lambda path, results, passed: None)

    args = cli.build_parser().parse_args(
        [
            "retrieval-eval",
            "--device", "cpu",
            "--query-batch-size", "7",
        ]
    )
    cli.cmd_retrieval_eval(args)

    assert captured["max_puuids"] == 123
    assert captured["model_device"] == "cpu"
    assert captured["state_dict"] == {"weight": "fake"}
    assert [run["holdout_label"] for run in captured["runs"]] == [
        "game_cold",
        "player_cold",
    ]
    assert all(run["device"] == "cpu" for run in captured["runs"])
    assert all(run["query_batch_size"] == 7 for run in captured["runs"])

    out = capsys.readouterr().out
    assert "[retrieval-eval] device=cpu cuda_available=True query_batch_size=7" in out


def test_retrieval_eval_parser_exposes_query_encoder_batch_size():
    from model.m4_eval import DEFAULT_QUERY_ENCODER_BATCH_SIZE

    defaults = cli.build_parser().parse_args(["retrieval-eval"])
    assert defaults.query_encoder_batch_size == 32
    # The CLI default and the library default must not drift apart.
    assert defaults.query_encoder_batch_size == DEFAULT_QUERY_ENCODER_BATCH_SIZE

    args = cli.build_parser().parse_args(
        ["retrieval-eval", "--query-encoder-batch-size", "9"]
    )
    assert args.query_encoder_batch_size == 9


def test_retrieval_eval_forwards_query_encoder_batch_size(monkeypatch, capsys):
    """cmd_retrieval_eval hands --query-encoder-batch-size to run_m4_eval.

    Deliberately a separate test from
    ``test_retrieval_eval_forwards_device_and_query_batch_size``: that one was
    already red before this change (its fake load_index returns a bare dict, so
    ``model_bundle.checkpoint_sha`` raises). This uses a bundle stub carrying
    checkpoint_sha so the forwarding is actually exercised.
    """
    from types import SimpleNamespace

    import model.m4_eval as m4_eval_module
    import model.plan_b_model as plan_b_model_module
    import model.retrieval as retrieval_module

    captured = {}

    class FakeModel:
        def __init__(self, max_puuids):
            pass

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            pass

    def fake_run_m4_eval(**kwargs):
        captured.setdefault("runs", []).append(kwargs)
        return {
            "n_queries": 1,
            "model": {
                "k_sweep": {64: 0.8},
                "per_minute_at_headline_k": {10: 0.8},
            },
        }

    monkeypatch.setattr(torch, "load", lambda *a, **kw: {
        "max_puuids": 123,
        "state_dict": {"weight": "fake"},
    })
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(plan_b_model_module, "PlanBModel", FakeModel)
    monkeypatch.setattr(
        retrieval_module, "load_index",
        lambda path: SimpleNamespace(checkpoint_sha="deadbeef"),
    )
    monkeypatch.setattr(m4_eval_module, "run_m4_eval", fake_run_m4_eval)
    monkeypatch.setattr(cli, "load_split", lambda name: {
        "train": ["train-1"],
        "holdout": ["game-cold-1"],
        "cold": ["player-cold-1"],
    }[name])
    monkeypatch.setattr(cli, "build_puuid_index", lambda ids, max_puuids: {})
    monkeypatch.setattr(cli, "_print_eval_result", lambda r: None)
    monkeypatch.setattr(cli, "_write_report", lambda path, results, passed: None)

    args = cli.build_parser().parse_args(
        ["retrieval-eval", "--query-encoder-batch-size", "8"]
    )
    cli.cmd_retrieval_eval(args)

    runs = captured["runs"]
    assert [run["holdout_label"] for run in runs] == ["game_cold", "player_cold"]
    assert all(run["query_encoder_batch_size"] == 8 for run in runs)
    assert "[retrieval-eval] query_encoder_batch_size=8" in capsys.readouterr().out


def test_retrieval_eval_rejects_non_positive_query_encoder_batch_size(monkeypatch):
    import model.plan_b_model as plan_b_model_module

    monkeypatch.setattr(torch, "load", lambda *a, **kw: {
        "max_puuids": 1, "state_dict": {},
    })
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(plan_b_model_module, "PlanBModel", lambda **kw: None)
    monkeypatch.setattr(cli, "load_split", lambda name: [])
    monkeypatch.setattr(cli, "build_puuid_index", lambda ids, max_puuids: {})

    args = cli.build_parser().parse_args(
        ["retrieval-eval", "--query-encoder-batch-size", "0"]
    )
    with pytest.raises(SystemExit, match="query-encoder-batch-size"):
        cli.cmd_retrieval_eval(args)
