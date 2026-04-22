"""Static game-context feature bundle for world-model inputs.

The canonical static stream mixes categorical game-start context (champion picks,
side, queue, region, time-of-day) with the existing numeric patch vector.
The vector layout is intentionally simple so dataset collation can keep using a
single float tensor, while the encoder can still interpret slices as typed
categorical fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import numpy as np

from raw_db import get_raw_match
from model.patch_params import PATCH_VECTOR_DIM, patch_vector_for_match

N_PARTICIPANTS: Final[int] = 10
STATIC_TOKEN_COUNT: Final[int] = N_PARTICIPANTS + 1
CHAMPION_ID_SLICE: Final[slice] = slice(0, N_PARTICIPANTS)
SIDE_SLICE: Final[slice] = slice(N_PARTICIPANTS, 2 * N_PARTICIPANTS)
QUEUE_INDEX: Final[int] = 2 * N_PARTICIPANTS
REGION_INDEX: Final[int] = QUEUE_INDEX + 1
TIME_BUCKET_INDEX: Final[int] = REGION_INDEX + 1
PATCH_SLICE: Final[slice] = slice(TIME_BUCKET_INDEX + 1, TIME_BUCKET_INDEX + 1 + PATCH_VECTOR_DIM)
STATIC_VECTOR_DIM: Final[int] = PATCH_SLICE.stop

MAX_CHAMPION_ID: Final[int] = 2048
MAX_QUEUE_ID: Final[int] = 2000
TIME_BUCKET_COUNT: Final[int] = 5  # 0 unknown + 4 coarse buckets

_PLATFORM_IDS = [
    "BR1", "EUN1", "EUW1", "JP1", "KR", "LA1", "LA2", "ME1", "NA1", "OC1", "PH2",
    "RU", "SG2", "TH2", "TR1", "TW2", "VN2",
]
PLATFORM_TO_REGION_ID = {platform: i + 1 for i, platform in enumerate(_PLATFORM_IDS)}
NUM_REGION_IDS: Final[int] = len(PLATFORM_TO_REGION_ID) + 1


def time_bucket_from_game_creation(game_creation_ms: int | None) -> int:
    if not game_creation_ms:
        return 0
    hour = datetime.fromtimestamp(game_creation_ms / 1000.0, tz=timezone.utc).hour
    if 0 <= hour < 6:
        return 1  # overnight
    if hour < 12:
        return 2  # morning
    if hour < 18:
        return 3  # afternoon
    return 4  # evening


def region_id_from_platform(platform_id: str | None) -> int:
    if not platform_id:
        return 0
    return PLATFORM_TO_REGION_ID.get(str(platform_id).upper(), 0)


def build_static_feature_vector(match_id: str) -> np.ndarray:
    """Return the canonical static feature vector for one match.

    Layout:
      [ champion_ids(10) | side_flags(10) | queue_id | region_id | time_bucket | patch_vector(256) ]
    """
    vec = np.zeros(STATIC_VECTOR_DIM, dtype=np.float32)

    match, _timeline = get_raw_match(match_id)
    if match is not None:
        participants = match.get("info", {}).get("participants", [])[:N_PARTICIPANTS]
        for i, participant in enumerate(participants):
            vec[CHAMPION_ID_SLICE.start + i] = float(participant.get("championId", 0) or 0)
            vec[SIDE_SLICE.start + i] = 1.0 if participant.get("teamId") == 200 else 0.0

        info = match.get("info", {})
        queue_id = int(info.get("queueId", 0) or 0)
        platform_id = info.get("platformId") or match.get("metadata", {}).get("platformId")
        vec[QUEUE_INDEX] = float(min(max(queue_id, 0), MAX_QUEUE_ID))
        vec[REGION_INDEX] = float(region_id_from_platform(platform_id))
        vec[TIME_BUCKET_INDEX] = float(time_bucket_from_game_creation(info.get("gameCreation")))

    vec[PATCH_SLICE] = patch_vector_for_match(match_id)
    return vec
