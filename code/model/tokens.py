# code/model/tokens.py
"""Fixed token vocabulary for the world-model input stream.

Meaningful events the spec calls out:
- Outcomes: CHAMPION_KILL, BUILDING_KILL, ELITE_MONSTER_KILL, CHAMPION_SPECIAL_KILL
- Decisions (explicit): ITEM_PURCHASED, SKILL_LEVEL_UP, WARD_PLACED, WARD_KILL
- Decisions (inferred): RECALL, ENGAGE, DISENGAGE

Reserved tokens occupy the first 8 IDs so event-type IDs start at 8.
"""

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
