import torch


FRAME_BASELINE_DIM = 91  # 9 stats × 10 participants + 1 minute


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
