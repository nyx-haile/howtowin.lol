from model.tokenizer import Token, tokenize_match
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
    gaps = [anchor_ts[i + 1] - anchor_ts[i] for i in range(len(anchor_ts) - 1)]
    assert all(abs(g - 60000) <= 1000 for g in gaps[:-1])


def test_event_tokens_reference_known_slots(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    kill_id = EVENT_TYPE_TO_ID["CHAMPION_KILL"]
    kills = [t for t in stream if t.type_id == kill_id]
    for k in kills:
        assert 1 <= k.actor_slot <= 10
        assert 0 <= k.target_slot <= 10


def test_payload_fields_surface_item_and_skill_info(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    items = [t for t in stream if t.type_id == EVENT_TYPE_TO_ID["ITEM_PURCHASED"]]
    skills = [t for t in stream if t.type_id == EVENT_TYPE_TO_ID["SKILL_LEVEL_UP"]]
    wards = [t for t in stream if t.type_id == EVENT_TYPE_TO_ID["WARD_PLACED"]]
    assert any(t.item_id > 0 for t in items)
    assert any(t.skill_slot > 0 for t in skills)
    if wards:
        assert any(t.ward_type_id >= 0 for t in wards)


def test_token_dataclass_fields():
    t = Token(type_id=0, actor_slot=0, target_slot=0, timestamp_ms=0)
    assert hasattr(t, "item_id") and hasattr(t, "skill_slot")
    assert hasattr(t, "monster_subtype_id") and hasattr(t, "ward_type_id")


def test_recall_tokens_inferred(fixture_match_id):
    stream = tokenize_match(fixture_match_id)
    recall_id = EVENT_TYPE_TO_ID["RECALL"]
    recalls = [t for t in stream if t.type_id == recall_id]
    assert len(recalls) >= 1
