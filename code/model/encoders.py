"""Three-stream encoders producing D_MODEL embeddings."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.player_features import PLAYER_FEATURE_DIM
from model.static_features import (
    CHAMPION_ID_SLICE,
    MAX_CHAMPION_ID,
    MAX_QUEUE_ID,
    N_PARTICIPANTS,
    NUM_REGION_IDS,
    PATCH_SLICE,
    QUEUE_INDEX,
    REGION_INDEX,
    SIDE_SLICE,
    STATIC_TOKEN_COUNT,
    STATIC_VECTOR_DIM,
    TIME_BUCKET_COUNT,
    TIME_BUCKET_INDEX,
)
from model.tokens import (
    MAX_ITEM_ID,
    NUM_BUILDING_TYPES,
    NUM_LANE_TYPES,
    NUM_MONSTER_SUBTYPES,
    NUM_MONSTER_TYPES,
    NUM_SLOTS,
    NUM_TOWER_TYPES,
    NUM_WARD_TYPES,
    SKILL_SLOT_COUNT,
    VOCAB_SIZE,
)

D_MODEL = 256


class StaticContextEncoder(nn.Module):
    def __init__(self, input_dim: int = STATIC_VECTOR_DIM, d_model: int = D_MODEL):
        super().__init__()
        if input_dim != STATIC_VECTOR_DIM:
            raise ValueError(f"StaticContextEncoder expects input_dim={STATIC_VECTOR_DIM}, got {input_dim}")
        self.champ_emb = nn.Embedding(MAX_CHAMPION_ID + 1, d_model, padding_idx=0)
        self.side_emb = nn.Embedding(2, d_model)
        self.slot_emb = nn.Embedding(N_PARTICIPANTS, d_model)
        self.queue_emb = nn.Embedding(MAX_QUEUE_ID + 1, d_model, padding_idx=0)
        self.region_emb = nn.Embedding(NUM_REGION_IDS, d_model, padding_idx=0)
        self.time_emb = nn.Embedding(TIME_BUCKET_COUNT, d_model, padding_idx=0)
        self.patch_proj = nn.Sequential(
            nn.Linear(PATCH_SLICE.stop - PATCH_SLICE.start, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pick_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.misc_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        champ_ids = x[:, CHAMPION_ID_SLICE].round().long().clamp(min=0, max=MAX_CHAMPION_ID)
        side_ids = x[:, SIDE_SLICE].round().long().clamp(min=0, max=1)
        queue_ids = x[:, QUEUE_INDEX].round().long().clamp(min=0, max=MAX_QUEUE_ID)
        region_ids = x[:, REGION_INDEX].round().long().clamp(min=0, max=NUM_REGION_IDS - 1)
        time_ids = x[:, TIME_BUCKET_INDEX].round().long().clamp(min=0, max=TIME_BUCKET_COUNT - 1)
        patch = x[:, PATCH_SLICE]

        slot_ids = torch.arange(N_PARTICIPANTS, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pick_tokens = (
            self.champ_emb(champ_ids)
            + self.side_emb(side_ids)
            + self.slot_emb(slot_ids)
        )
        pick_tokens = self.pick_net(pick_tokens)

        misc_token = (
            self.queue_emb(queue_ids)
            + self.region_emb(region_ids)
            + self.time_emb(time_ids)
            + self.patch_proj(patch)
        )
        misc_token = self.misc_net(misc_token).unsqueeze(1)

        tokens = torch.cat([pick_tokens, misc_token], dim=1)
        if tokens.size(1) != STATIC_TOKEN_COUNT:
            raise RuntimeError(f"expected {STATIC_TOKEN_COUNT} static tokens, got {tokens.size(1)}")
        return tokens


class PlayerModelEncoder(nn.Module):
    def __init__(self, max_puuids, feature_dim=PLAYER_FEATURE_DIM, d_model=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.residual = nn.Embedding(max_puuids, d_model, padding_idx=0)
        nn.init.zeros_(self.residual.weight[0])

    def forward(self, crafted, puuid_ids):
        base = self.mlp(crafted)
        delta = self.residual(puuid_ids)
        return base + delta


class DynamicStreamEmbedder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_slots=NUM_SLOTS, d_model=D_MODEL):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.actor_emb = nn.Embedding(num_slots, d_model, padding_idx=0)
        self.target_emb = nn.Embedding(num_slots, d_model, padding_idx=0)
        self.item_emb = nn.Embedding(MAX_ITEM_ID + 1, d_model, padding_idx=0)
        self.skill_emb = nn.Embedding(SKILL_SLOT_COUNT, d_model, padding_idx=0)
        self.monster_type_emb = nn.Embedding(NUM_MONSTER_TYPES, d_model, padding_idx=0)
        self.monster_subtype_emb = nn.Embedding(NUM_MONSTER_SUBTYPES, d_model, padding_idx=0)
        self.building_type_emb = nn.Embedding(NUM_BUILDING_TYPES, d_model, padding_idx=0)
        self.lane_type_emb = nn.Embedding(NUM_LANE_TYPES, d_model, padding_idx=0)
        self.tower_type_emb = nn.Embedding(NUM_TOWER_TYPES, d_model, padding_idx=0)
        self.ward_type_emb = nn.Embedding(NUM_WARD_TYPES, d_model, padding_idx=0)
        self.d_model = d_model

    def _positional(self, ts):
        B, L = ts.shape
        device = ts.device
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        t = (ts / 60000.0).unsqueeze(-1)
        args = t * div
        pe = torch.zeros(B, L, self.d_model, device=device)
        pe[..., 0::2] = torch.sin(args)
        pe[..., 1::2] = torch.cos(args)
        return pe

    def _optional_long(self, maybe_tensor, ref):
        if maybe_tensor is None:
            return torch.zeros_like(ref, dtype=torch.long)
        return maybe_tensor.long()

    def forward(
        self,
        tokens,
        actors,
        timestamps,
        player_emb,
        targets=None,
        item_ids=None,
        skill_slots=None,
        monster_types=None,
        monster_subtypes=None,
        building_types=None,
        lane_types=None,
        tower_types=None,
        ward_types=None,
    ):
        tok = self.token_emb(tokens)
        act = self.actor_emb(actors)
        pos = self._positional(timestamps)

        targets = self._optional_long(targets, actors).clamp(min=0, max=NUM_SLOTS - 1)
        item_ids = self._optional_long(item_ids, actors).clamp(min=0, max=MAX_ITEM_ID)
        skill_slots = self._optional_long(skill_slots, actors).clamp(min=0, max=SKILL_SLOT_COUNT - 1)
        monster_types = self._optional_long(monster_types, actors).clamp(min=0, max=NUM_MONSTER_TYPES - 1)
        monster_subtypes = self._optional_long(monster_subtypes, actors).clamp(min=0, max=NUM_MONSTER_SUBTYPES - 1)
        building_types = self._optional_long(building_types, actors).clamp(min=0, max=NUM_BUILDING_TYPES - 1)
        lane_types = self._optional_long(lane_types, actors).clamp(min=0, max=NUM_LANE_TYPES - 1)
        tower_types = self._optional_long(tower_types, actors).clamp(min=0, max=NUM_TOWER_TYPES - 1)
        ward_types = self._optional_long(ward_types, actors).clamp(min=0, max=NUM_WARD_TYPES - 1)

        payload = (
            self.target_emb(targets)
            + self.item_emb(item_ids)
            + self.skill_emb(skill_slots)
            + self.monster_type_emb(monster_types)
            + self.monster_subtype_emb(monster_subtypes)
            + self.building_type_emb(building_types)
            + self.lane_type_emb(lane_types)
            + self.tower_type_emb(tower_types)
            + self.ward_type_emb(ward_types)
        )

        B, L, D = tok.shape
        actor_mask = actors > 0
        actor_safe = actors.clamp(min=1) - 1
        player_gather = torch.gather(
            player_emb,
            1,
            actor_safe.unsqueeze(-1).expand(B, L, D),
        )
        player_gather = player_gather * actor_mask.unsqueeze(-1).to(player_gather.dtype)

        return tok + act + payload + pos + player_gather
