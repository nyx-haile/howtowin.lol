"""Coarse rank-band contract shared by diagnostics and skill-aware retrieval.

Ties together three things:
  - the 4-band scheme (iron_silver, gold_platinum, emerald_diamond, master_plus),
  - per-player band derivation from the `player_features` rank-tier column,
  - per-game band derivation (mode of ranked players' bands).

Kept narrow on purpose — no torch.nn, no sklearn, no db access. Importable from
retrieval.py without pulling the full diagnostics surface.
"""
from __future__ import annotations

import torch

TIER_TO_BAND: dict[int, int] = {
    0: -1,
    1: 0, 2: 0, 3: 0,           # Iron, Bronze, Silver
    4: 1, 5: 1,                 # Gold, Platinum
    6: 2, 7: 2,                 # Emerald, Diamond
    8: 3, 9: 3, 10: 3,          # Master, Grandmaster, Challenger
}
BAND_NAMES = ["iron_silver", "gold_platinum", "emerald_diamond", "master_plus"]
N_BANDS = 4
UNRANKED_BAND = -1

BAND_TIER_NAMES: dict[int, tuple[str, ...]] = {
    0: ("IRON", "BRONZE", "SILVER"),
    1: ("GOLD", "PLATINUM"),
    2: ("EMERALD", "DIAMOND"),
    3: ("MASTER", "GRANDMASTER", "CHALLENGER"),
}

MIN_RANKED_PLAYERS_FOR_GAME_BAND = 1


def tier_to_band_tensor(tiers: torch.Tensor) -> torch.Tensor:
    """Map tier ordinal (0..10) to coarse band id (0..3) or -1 (unranked)."""
    lut = torch.tensor(
        [-1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
        dtype=torch.long,
        device=tiers.device,
    )
    t = tiers.round().long().clamp(min=0, max=10)
    band = lut[t]
    band[tiers <= 0] = -1
    return band


def player_band(players: torch.Tensor) -> torch.Tensor:
    """Per-player band. ``players``: ``(..., 10, PLAYER_FEATURE_DIM)``; col 0 is rank_tier_ordinal.

    Returns same leading shape ``(..., 10)`` long, values in ``{0..3}`` or ``-1``.
    """
    tiers = players[..., 0]
    return tier_to_band_tensor(tiers)


def game_band(players: torch.Tensor) -> torch.Tensor:
    """Per-game rank band: mode of ranked players' coarse bands.

    ``players``: ``(B, 10, PLAYER_FEATURE_DIM)``.
    Returns ``(B,)`` long; value in ``{0..3}`` if at least
    ``MIN_RANKED_PLAYERS_FOR_GAME_BAND`` players are ranked, else ``-1``.

    In the 51k corpus "unranked" usually means backfill pending; matchmaking
    binds a lobby tightly enough that even one ranked player's band is a
    faithful proxy for the lobby band.
    """
    bands = player_band(players)   # (B, 10) long
    B = bands.shape[0]
    out = torch.full((B,), -1, dtype=torch.long, device=bands.device)
    for b in range(B):
        ranked = bands[b][bands[b] >= 0]
        if ranked.numel() < MIN_RANKED_PLAYERS_FOR_GAME_BAND:
            continue
        vals, counts = torch.unique(ranked, return_counts=True)
        out[b] = vals[counts.argmax()]
    return out


def band_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Integer ordinal distance between two band ids in ``{-1, 0..3}``.

    Unranked (-1) on either side yields a sentinel distance of N_BANDS so it
    compares "further" than any valid cross-band pair. Callers that want to
    treat unranked as wildcard must handle it before calling this.
    """
    a_valid = a >= 0
    b_valid = b >= 0
    diff = (a - b).abs()
    sentinel = torch.full_like(diff, N_BANDS)
    return torch.where(a_valid & b_valid, diff, sentinel)
