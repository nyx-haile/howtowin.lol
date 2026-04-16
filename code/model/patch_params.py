"""Data Dragon cache + per-game patch-parameter vector.

Pulls item.json and champion.json from Community Dragon CDN, caches locally
under data/dragon_cache/<patch>/. The per-game vector is a fixed-length
dense summary: champion stats (base stats + Q/W/E/R info rank 1-5 for
picks present) + aggregated item stats present in the game. The exact
dimension is controlled by PATCH_VECTOR_DIM and the fields enumerated
below; keep stable across Plan A so the encoder layer has a fixed input.
"""
import json
import os
import requests
import numpy as np

from db import get_conn
from model.patch_modes import (
    PATCH_VECTOR_MODE_SCAFFOLDING,
    PATCH_VECTOR_MODE_FULL,
    DEFAULT_PATCH_VECTOR_MODE,
)

DRAGON_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dragon_cache')
DDRAGON_BASE = "https://ddragon.leagueoflegends.com/cdn"

# Dimension breakdown (total 1024; Plan B §A3):
#   10 picks * 10 stat fields = 100  (base stats per champion present)
#   Aggregated item-stat totals across all items on the patch = 20
#   Version one-hot proxy (major, minor, point) = 3 (offsets 120..122)
#   Reserved zeros for Milestone 6 schema expansion = remainder
PATCH_VECTOR_DIM = 1024

CHAMP_STAT_FIELDS = ["hp", "hpperlevel", "mp", "mpperlevel", "armor",
                     "armorperlevel", "attackdamage", "attackdamageperlevel",
                     "attackspeedoffset", "attackspeed"]

ITEM_STAT_FIELDS = ["FlatPhysicalDamageMod", "FlatMagicDamageMod",
                    "FlatArmorMod", "FlatSpellBlockMod", "FlatHPPoolMod",
                    "FlatMPPoolMod", "PercentAttackSpeedMod",
                    "FlatCritChanceMod", "FlatMovementSpeedMod", "FlatHPRegenMod",
                    "PercentLifeStealMod", "FlatBlockMod",
                    "FlatEnergyPoolMod", "FlatMagicPenetrationMod",
                    "FlatPhysicalDamageReduction", "FlatArmorPenetrationMod",
                    "PercentMovementSpeedMod", "PercentArmorMod",
                    "PercentMagicPenetrationMod", "PercentCritChanceMod"]


def fetch_dragon_version(patch):
    """Given a match's gameVersion like '14.14.580.1234', return a Data Dragon
    patch id like '14.14.1'. For Plan A, default to '14.14.1' when match
    patches don't resolve cleanly."""
    if not patch:
        return "14.14.1"
    parts = patch.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.1"
    return "14.14.1"


def _cache_path(version, filename):
    return os.path.join(DRAGON_CACHE_DIR, version, filename)


def _download(version, filename):
    url = f"{DDRAGON_BASE}/{version}/data/en_US/{filename}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path = _cache_path(version, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(resp.text)
    return resp.json()


def _load_json(version, filename):
    path = _cache_path(version, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return _download(version, filename)


def load_items(version):
    return _load_json(version, "item.json")["data"]


def load_champions(version):
    raw = _load_json(version, "champion.json")["data"]
    # champion.json summaries are keyed by champ name; stats are nested.
    return {cname: cdata for cname, cdata in raw.items()}


def _champ_name_by_key(champs, key):
    """Data Dragon uses champ names as keys; Riot match API uses numeric
    championId ('key' field). Build reverse index."""
    for name, data in champs.items():
        if str(data.get("key")) == str(key):
            return name
    return None


def patch_vector_for_match(match_id, mode=DEFAULT_PATCH_VECTOR_MODE):
    conn = get_conn()
    try:
        game = conn.execute(
            "SELECT patch FROM games WHERE match_id = ?", (match_id,)
        ).fetchone()
    finally:
        conn.close()

    version = fetch_dragon_version(game["patch"] if game else "")
    champs = load_champions(version)
    items = load_items(version)

    vec = np.zeros(PATCH_VECTOR_DIM, dtype=np.float32)

    if mode == PATCH_VECTOR_MODE_FULL:
        # Champion stats per participant.
        from raw_db import get_raw_match
        match, _tl = get_raw_match(match_id)
        if match:
            for i, p in enumerate(match["info"]["participants"][:10]):
                champ_key = p.get("championId")
                champ_name = _champ_name_by_key(champs, champ_key)
                if champ_name and champ_name in champs:
                    stats = champs[champ_name].get("stats", {})
                    for j, fname in enumerate(CHAMP_STAT_FIELDS):
                        vec[i * len(CHAMP_STAT_FIELDS) + j] = float(stats.get(fname, 0.0))

        # Patch-level item aggregate (match-independent).
        item_totals = {f: 0.0 for f in ITEM_STAT_FIELDS}
        for idata in items.values():
            stats = idata.get("stats", {})
            for fname in ITEM_STAT_FIELDS:
                item_totals[fname] += float(stats.get(fname, 0.0))
        for j, fname in enumerate(ITEM_STAT_FIELDS):
            vec[100 + j] = item_totals[fname]

    # Version triple — always populated, in both modes.
    parts = version.split(".")
    vec[120] = float(parts[0]) if parts and parts[0].isdigit() else 0.0
    vec[121] = float(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0.0
    vec[122] = float(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0.0

    return vec
