# code/model/tokens.py
"""Token vocabulary + categorical payload vocabularies for the world model."""
from __future__ import annotations

ANCHOR_TOKEN = 0
PAD_TOKEN = 1
BOS_TOKEN = 2
_RESERVED_COUNT = 8

EVENT_TYPES = [
    "CHAMPION_KILL",
    "BUILDING_KILL",
    "ELITE_MONSTER_KILL",
    "CHAMPION_SPECIAL_KILL",
    "ITEM_PURCHASED",
    "SKILL_LEVEL_UP",
    "WARD_PLACED",
    "WARD_KILL",
    "RECALL",
    "ENGAGE",
    "DISENGAGE",
]

EVENT_TYPE_TO_ID = {name: _RESERVED_COUNT + i for i, name in enumerate(EVENT_TYPES)}
ID_TO_EVENT_TYPE = {i: name for name, i in EVENT_TYPE_TO_ID.items()}

NUM_EVENT_TYPES = len(EVENT_TYPES)
NUM_SLOTS = 11  # slot 0 = world/none; slots 1..10 = participants
VOCAB_SIZE = _RESERVED_COUNT + NUM_EVENT_TYPES

NO_EVENT_LABEL = 0
EVENT_TYPE_LABEL_COUNT = NUM_EVENT_TYPES + 1  # +1 for "no next event"

MAX_ITEM_ID = 10000
SKILL_SLOT_COUNT = 5  # 0 none, 1..4 => Q/W/E/R

MONSTER_TYPE_TO_ID = {
    None: 0,
    "NONE": 0,
    "UNKNOWN": 1,
    "DRAGON": 2,
    "BARON_NASHOR": 3,
    "RIFTHERALD": 4,
    "HORDE": 5,
    "ATAKHAN": 6,
}
MONSTER_SUBTYPE_TO_ID = {
    None: 0,
    "NONE": 0,
    "UNKNOWN": 1,
    "AIR_DRAGON": 2,
    "CHEMTECH_DRAGON": 3,
    "EARTH_DRAGON": 4,
    "FIRE_DRAGON": 5,
    "HEXTECH_DRAGON": 6,
    "WATER_DRAGON": 7,
    "ELDER_DRAGON": 8,
    "RIFTHERALD": 9,
    "BARON_NASHOR": 10,
    "HORDE": 11,
    "ATAKHAN": 12,
}
BUILDING_TYPE_TO_ID = {
    None: 0,
    "NONE": 0,
    "UNKNOWN": 1,
    "TOWER_BUILDING": 2,
    "INHIBITOR_BUILDING": 3,
    "NEXUS_BUILDING": 4,
}
LANE_TYPE_TO_ID = {
    None: 0,
    "NONE": 0,
    "UNKNOWN": 1,
    "TOP_LANE": 2,
    "MID_LANE": 3,
    "BOT_LANE": 4,
}
TOWER_TYPE_TO_ID = {
    None: 0,
    "NONE": 0,
    "UNKNOWN": 1,
    "OUTER_TURRET": 2,
    "INNER_TURRET": 3,
    "BASE_TURRET": 4,
    "NEXUS_TURRET": 5,
}
WARD_TYPE_TO_ID = {
    None: 0,
    "NONE": 0,
    "UNKNOWN": 1,
    "YELLOW_TRINKET": 2,
    "CONTROL_WARD": 3,
    "BLUE_TRINKET": 4,
    "SIGHT_WARD": 5,
    "UNDEFINED": 6,
}

NUM_MONSTER_TYPES = max(MONSTER_TYPE_TO_ID.values()) + 1
NUM_MONSTER_SUBTYPES = max(MONSTER_SUBTYPE_TO_ID.values()) + 1
NUM_BUILDING_TYPES = max(BUILDING_TYPE_TO_ID.values()) + 1
NUM_LANE_TYPES = max(LANE_TYPE_TO_ID.values()) + 1
NUM_TOWER_TYPES = max(TOWER_TYPE_TO_ID.values()) + 1
NUM_WARD_TYPES = max(WARD_TYPE_TO_ID.values()) + 1


def lookup_with_unknown(mapping: dict, raw_value) -> int:
    if raw_value in mapping:
        return mapping[raw_value]
    key = str(raw_value).upper() if raw_value is not None else None
    if key in mapping:
        return mapping[key]
    return mapping.get("UNKNOWN", 0)


def event_label_from_type_id(type_id: int) -> int:
    """Map token type id -> training label id with 0 reserved for no-event."""
    if type_id < _RESERVED_COUNT:
        return NO_EVENT_LABEL
    return (type_id - _RESERVED_COUNT) + 1


def event_label_to_token_type(label: int) -> int:
    if label <= 0:
        raise ValueError("label 0 is reserved for no-event")
    return _RESERVED_COUNT + (label - 1)
