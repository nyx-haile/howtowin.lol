"""Three-stream encoders producing D_MODEL embeddings.

StaticContextEncoder: MLP over dense patch-param + game-context vector.
PlayerModelEncoder:   crafted-features MLP + per-puuid residual embedding.
DynamicStreamEmbedder: token-type + actor-slot + player-fusion + timestamp
                      positional encoding.

The residual embedding table has index 0 reserved for "unknown puuid"
(zero-init, frozen to zero during training).
"""
import math
import torch
import torch.nn as nn

from model.tokens import VOCAB_SIZE, NUM_SLOTS
from model.patch_params import PATCH_VECTOR_DIM
from model.player_features import PLAYER_FEATURE_DIM

D_MODEL = 256


class StaticContextEncoder(nn.Module):
    def __init__(self, input_dim=PATCH_VECTOR_DIM, d_model=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class PlayerModelEncoder(nn.Module):
    def __init__(self, max_puuids, feature_dim=PLAYER_FEATURE_DIM, d_model=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Index 0 is "unknown" (frozen-zero); 1..max_puuids-1 are learnable.
        self.residual = nn.Embedding(max_puuids, d_model, padding_idx=0)
        nn.init.zeros_(self.residual.weight[0])

    def forward(self, crafted, puuid_ids):
        """crafted: (B, 10, feature_dim). puuid_ids: (B, 10)."""
        base = self.mlp(crafted)
        delta = self.residual(puuid_ids)
        return base + delta


class DynamicStreamEmbedder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_slots=NUM_SLOTS, d_model=D_MODEL):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.actor_emb = nn.Embedding(num_slots, d_model)
        self.d_model = d_model

    def _positional(self, ts):
        """Sinusoidal positional encoding from real timestamps (ms)."""
        B, L = ts.shape
        device = ts.device
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        # Normalize timestamps to minutes so sin args are reasonable.
        t = (ts / 60000.0).unsqueeze(-1)
        args = t * div
        pe = torch.zeros(B, L, self.d_model, device=device)
        pe[..., 0::2] = torch.sin(args)
        pe[..., 1::2] = torch.cos(args)
        return pe

    def forward(self, tokens, actors, timestamps, player_emb):
        """tokens, actors: (B, L). timestamps: (B, L). player_emb: (B, 10, D)."""
        tok = self.token_emb(tokens)
        act = self.actor_emb(actors)
        pos = self._positional(timestamps)

        # Fuse player embedding at event positions whose actor is 1..10.
        B, L, D = tok.shape
        # Gather per-token player embedding (zero for actor_slot == 0).
        actor_mask = (actors > 0)  # (B, L)
        actor_safe = actors.clamp(min=1) - 1  # (B, L) in [0, 9]
        player_gather = torch.gather(
            player_emb, 1,
            actor_safe.unsqueeze(-1).expand(B, L, D)
        )
        player_gather = player_gather * actor_mask.unsqueeze(-1).to(player_gather.dtype)

        return tok + act + pos + player_gather
