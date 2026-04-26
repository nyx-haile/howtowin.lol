import torch
import pytest


FRAME_BASELINE_DIM = 91  # 9 stats × 10 participants + 1 minute


def _first_distinct_match_ids(all_match_ids, count=2):
    out = []
    for mid in all_match_ids:
        if mid not in out:
            out.append(mid)
        if len(out) == count:
            return out
    pytest.skip(f"need at least {count} distinct fixture matches")


def test_frame_features_index_dim_is_91(fixture_match_id):
    from model.baselines.frame_features_index import build_frame_features_index
    bundle = build_frame_features_index(
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
    )
    assert bundle.corpus_white.shape[1] == FRAME_BASELINE_DIM


def test_frame_features_index_excludes(fixture_match_id):
    from model.baselines.frame_features_index import build_frame_features_index
    bundle = build_frame_features_index(
        train_match_ids=[fixture_match_id],
        exclude_match_ids={fixture_match_id},
    )
    assert bundle.corpus_white.shape[0] == 0


def test_frame_features_index_records_anchor_minutes(fixture_match_id):
    from model.baselines.frame_features_index import build_frame_features_index
    bundle = build_frame_features_index(
        train_match_ids=[fixture_match_id],
        exclude_match_ids=set(),
    )
    N = bundle.corpus_white.shape[0]
    assert bundle.row_anchor_minute.shape == (N,)
    assert bundle.row_blue_win.shape == (N,)
    # All from one match.
    assert all(mid == fixture_match_id for mid in bundle.row_match_id)


def test_frame_features_index_batched_db_matches_per_match_order(all_match_ids):
    from model.baselines.frame_features_index import build_frame_features_index
    match_ids = _first_distinct_match_ids(all_match_ids, count=2)

    per_match = build_frame_features_index(
        train_match_ids=match_ids,
        exclude_match_ids=set(),
        db_chunk_size=1,
    )
    batched = build_frame_features_index(
        train_match_ids=match_ids,
        exclude_match_ids=set(),
        db_chunk_size=100,
    )

    assert batched.row_match_id == per_match.row_match_id
    assert torch.equal(batched.row_anchor_minute, per_match.row_anchor_minute)
    assert torch.equal(batched.row_blue_win, per_match.row_blue_win)
    assert torch.allclose(batched.corpus_white, per_match.corpus_white, atol=1e-6)


def test_frame_features_index_uses_one_connection_per_chunk(monkeypatch, all_match_ids):
    from model.baselines import frame_features_index as ffi
    match_ids = _first_distinct_match_ids(all_match_ids, count=2)
    calls = 0
    real_get_conn = ffi.get_conn

    def counting_get_conn(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_get_conn(*args, **kwargs)

    monkeypatch.setattr(ffi, "get_conn", counting_get_conn)
    ffi.build_frame_features_index(
        train_match_ids=match_ids,
        exclude_match_ids=set(),
        db_chunk_size=100,
    )

    assert calls == 1


def test_read_game_frames_many_preserves_ordering(all_match_ids):
    from model.baselines.frame_features_index import read_game_frames_many
    match_ids = _first_distinct_match_ids(all_match_ids, count=2)
    grouped = read_game_frames_many(match_ids, chunk_size=100)

    assert list(grouped) == match_ids
    for mid in match_ids:
        rows, _ = grouped[mid]
        row_order = [(r["timestamp_ms"], r["participant_slot"]) for r in rows]
        assert row_order == sorted(row_order)
