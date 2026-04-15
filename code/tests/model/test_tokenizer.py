from model.tokenizer import tokenize_match, Token
from model.tokens import ANCHOR_TOKEN, EVENT_TYPE_TO_ID


def test_tokenize_returns_non_empty_stream(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    assert len(stream) > 0


def test_stream_starts_with_anchor_at_t0(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    first = stream[0]
    assert first.type_id == ANCHOR_TOKEN
    assert first.timestamp_ms == 0


def test_stream_sorted_by_timestamp(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    timestamps = [t.timestamp_ms for t in stream]
    assert timestamps == sorted(timestamps)


def test_anchor_interval_is_60s(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    anchor_ts = [t.timestamp_ms for t in stream if t.type_id == ANCHOR_TOKEN]
    assert len(anchor_ts) >= 2
    # Consecutive anchor gaps should be 60000ms (Riot timeline cadence).
    # The very last frame is emitted at match end rather than on the minute
    # boundary, so the final gap may be short; exclude it from the tolerance check.
    gaps = [anchor_ts[i + 1] - anchor_ts[i] for i in range(len(anchor_ts) - 1)]
    assert all(abs(g - 60000) <= 1000 for g in gaps[:-1])


def test_event_tokens_reference_known_slots(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    kill_id = EVENT_TYPE_TO_ID["CHAMPION_KILL"]
    kills = [t for t in stream if t.type_id == kill_id]
    for k in kills:
        assert 1 <= k.actor_slot <= 10
        assert 0 <= k.target_slot <= 10


def test_token_dataclass_fields():
    t = Token(type_id=0, actor_slot=0, target_slot=0, timestamp_ms=0)
    assert hasattr(t, "type_id") and hasattr(t, "actor_slot")
    assert hasattr(t, "target_slot") and hasattr(t, "timestamp_ms")


def test_recall_tokens_inferred(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    recall_id = EVENT_TYPE_TO_ID["RECALL"]
    recalls = [t for t in stream if t.type_id == recall_id]
    # A 30+ minute game always has recalls.
    assert len(recalls) >= 1
