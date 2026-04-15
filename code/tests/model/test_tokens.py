from model.tokens import (
    EVENT_TYPES, EVENT_TYPE_TO_ID, ID_TO_EVENT_TYPE,
    ANCHOR_TOKEN, PAD_TOKEN, BOS_TOKEN,
    NUM_SLOTS, NUM_EVENT_TYPES,
)


def test_reserved_tokens_are_distinct():
    assert len({ANCHOR_TOKEN, PAD_TOKEN, BOS_TOKEN}) == 3


def test_event_types_cover_spec():
    required = {
        "CHAMPION_KILL", "BUILDING_KILL", "ELITE_MONSTER_KILL",
        "ITEM_PURCHASED", "SKILL_LEVEL_UP",
        "WARD_PLACED", "WARD_KILL", "CHAMPION_SPECIAL_KILL",
        "RECALL", "ENGAGE", "DISENGAGE",
    }
    assert required.issubset(set(EVENT_TYPES))


def test_bidirectional_mapping():
    for name in EVENT_TYPES:
        i = EVENT_TYPE_TO_ID[name]
        assert ID_TO_EVENT_TYPE[i] == name


def test_slot_range():
    assert NUM_SLOTS == 11  # slots 0 (world) + 1..10 (participants)


def test_num_event_types_matches_list():
    assert NUM_EVENT_TYPES == len(EVENT_TYPES)
