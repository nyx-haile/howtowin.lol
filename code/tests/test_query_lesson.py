"""Cache + error behaviour of the `lesson` action in code/query.py.

model.lesson / model.retrieval are stubbed so this runs without torch,
a checkpoint, or the retrieval index.
"""
import json
import sys
import types

import pytest

import query


@pytest.fixture
def stub_model(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt.pt"
    index = tmp_path / "index.pt"
    ckpt.write_bytes(b"c")
    index.write_bytes(b"i")
    calls = []

    def generate_lesson(match_id, team, k, skill_aware):
        calls.append((match_id, team, k))
        print("progress noise")  # must not corrupt stdout JSON
        return {"match_id": match_id, "team": team, "n_mid_anchors": 3}

    lesson_mod = types.ModuleType("model.lesson")
    lesson_mod.DEFAULT_CHECKPOINT_PATH = str(ckpt)
    lesson_mod.generate_lesson = generate_lesson
    lesson_mod.lesson_to_dict = lambda r: r
    retrieval_mod = types.ModuleType("model.retrieval")
    retrieval_mod.DEFAULT_INDEX_PATH = str(index)
    monkeypatch.setitem(sys.modules, "model.lesson", lesson_mod)
    monkeypatch.setitem(sys.modules, "model.retrieval", retrieval_mod)
    monkeypatch.setattr(query, "LESSON_CACHE_DIR", str(tmp_path / "cache"))
    return types.SimpleNamespace(ckpt=ckpt, index=index, calls=calls)


def test_lesson_is_cached(stub_model):
    a = query.get_lesson("NA1_1", "red")
    b = query.get_lesson("NA1_1", "red")
    assert a == b == {"match_id": "NA1_1", "team": "red", "n_mid_anchors": 3}
    assert len(stub_model.calls) == 1


def test_cache_invalidates_when_checkpoint_changes(stub_model):
    import os
    query.get_lesson("NA1_1", "blue")
    stub_model.ckpt.write_bytes(b"retrained-and-bigger")
    os.utime(stub_model.ckpt, (1, 1))
    query.get_lesson("NA1_1", "blue")
    assert len(stub_model.calls) == 2


def test_force_bypasses_cache(stub_model):
    query.get_lesson("NA1_1", "blue")
    query.get_lesson("NA1_1", "blue", force=True)
    assert len(stub_model.calls) == 2


def test_missing_checkpoint_reports_model_unavailable(stub_model):
    stub_model.ckpt.unlink()
    out = query.get_lesson("NA1_1", "blue")
    assert out["error"] == "model_unavailable"
    assert stub_model.calls == []


@pytest.mark.parametrize("match_id,team", [("NA1_1", "green"), ("../etc", "blue"), ("", "blue")])
def test_rejects_bad_input(stub_model, match_id, team):
    assert "error" in query.get_lesson(match_id, team)


def test_main_keeps_stdout_pure_json(stub_model, monkeypatch, capsys):
    import io
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"action": "lesson", "match_id": "NA1_2", "team": "blue"})))
    query.main()
    out = capsys.readouterr().out
    assert json.loads(out)["match_id"] == "NA1_2"
