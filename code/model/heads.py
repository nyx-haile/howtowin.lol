from __future__ import annotations

import torch
import torch.nn as nn

from model.tokens import (
    EVENT_TYPE_LABEL_COUNT,
    MAX_ITEM_ID,
    NUM_BUILDING_TYPES,
    NUM_LANE_TYPES,
    NUM_MONSTER_SUBTYPES,
    NUM_MONSTER_TYPES,
    NUM_SLOTS,
    NUM_TOWER_TYPES,
    NUM_WARD_TYPES,
    SKILL_SLOT_COUNT,
)


class FactorizedNextEventHead(nn.Module):
    """Factorized next-event head.

    Returns a dict of logits keyed by semantic factor rather than a single flat
    event identity.
    """

    def __init__(self, d_in: int, d_hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
        )
        self.type_head = nn.Linear(d_hidden, EVENT_TYPE_LABEL_COUNT)
        self.actor_head = nn.Linear(d_hidden, NUM_SLOTS)
        self.target_head = nn.Linear(d_hidden, NUM_SLOTS)
        self.item_head = nn.Linear(d_hidden, MAX_ITEM_ID + 1)
        self.skill_head = nn.Linear(d_hidden, SKILL_SLOT_COUNT)
        self.monster_type_head = nn.Linear(d_hidden, NUM_MONSTER_TYPES)
        self.monster_subtype_head = nn.Linear(d_hidden, NUM_MONSTER_SUBTYPES)
        self.building_type_head = nn.Linear(d_hidden, NUM_BUILDING_TYPES)
        self.lane_type_head = nn.Linear(d_hidden, NUM_LANE_TYPES)
        self.tower_type_head = nn.Linear(d_hidden, NUM_TOWER_TYPES)
        self.ward_type_head = nn.Linear(d_hidden, NUM_WARD_TYPES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "type_logits": self.type_head(h),
            "actor_logits": self.actor_head(h),
            "target_logits": self.target_head(h),
            "item_logits": self.item_head(h),
            "skill_logits": self.skill_head(h),
            "monster_type_logits": self.monster_type_head(h),
            "monster_subtype_logits": self.monster_subtype_head(h),
            "building_type_logits": self.building_type_head(h),
            "lane_type_logits": self.lane_type_head(h),
            "tower_type_logits": self.tower_type_head(h),
            "ward_type_logits": self.ward_type_head(h),
        }


class OutcomeHead(nn.Module):
    """Scalar logit for game-win at each anchor."""

    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NextDecisionHead(nn.Module):
    """Per-participant categorical over decision types."""

    def __init__(self, d_in: int, n_decisions: int, n_participants: int = 10,
                 d_hidden: int = 256):
        super().__init__()
        self.n_participants = n_participants
        self.n_decisions = n_decisions
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_participants * n_decisions),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), self.n_participants, self.n_decisions)


class NextFrameHead(nn.Module):
    """Per-participant mu/logvar over frame feature deltas."""

    def __init__(self, d_in: int, n_participants: int = 10, feat_dim: int = 6,
                 d_hidden: int = 256):
        super().__init__()
        self.n_participants = n_participants
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * n_participants * feat_dim),
        )

    def forward(self, x):
        out = self.net(x).view(x.size(0), self.n_participants, self.feat_dim, 2)
        mu = out[..., 0]
        logvar = out[..., 1].clamp(min=-10.0, max=10.0)
        return mu, logvar
