"""CausalTransformerBaseline: predicts multi-hot next-minute event set at
each anchor position.

Architecture:
  - StaticContextEncoder produces a small (B, S, D) static token set prepended as prefix tokens.
  - PlayerModelEncoder produces per-participant (B, 10, D) used by
    DynamicStreamEmbedder to fuse player identity into actor tokens.
  - Causal Transformer over [static_ctx_tokens, dynamic_stream].
  - Next-event head reads at each dynamic position -> (B, L, NUM_EVENT_TYPES).
"""
import torch
import torch.nn as nn

from model.encoders import (
    StaticContextEncoder, PlayerModelEncoder, DynamicStreamEmbedder, D_MODEL,
)
from model.tokens import NUM_EVENT_TYPES


class CausalTransformerBaseline(nn.Module):
    def __init__(self, max_puuids, n_layers=6, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.static_enc = StaticContextEncoder()
        self.player_enc = PlayerModelEncoder(max_puuids=max_puuids)
        self.dyn_emb = DynamicStreamEmbedder()

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.head = nn.Linear(D_MODEL, NUM_EVENT_TYPES)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, batch):
        static = self.static_enc(batch["static"])              # (B, S, D)
        players = self.player_enc(batch["players"], batch["player_ids"])  # (B, 10, D)
        dyn = self.dyn_emb(
            batch["tokens"], batch["token_actors"],
            batch["token_timestamps"], players,
        )  # (B, L, D)

        B, L, D = dyn.shape
        static_len = static.size(1)
        seq = torch.cat([static, dyn], dim=1)  # (B, S+L, D)

        # Causal mask over seq; static prefix tokens are always visible.
        mask = self._causal_mask(static_len + L, seq.device)
        mask[:static_len, :static_len] = False

        # Key-padding mask: static tokens are never padded; dynamic uses key_pad_mask.
        key_pad_mask = batch.get("key_pad_mask")
        if key_pad_mask is not None:
            pad = torch.cat([
                torch.zeros(B, static_len, dtype=torch.bool, device=seq.device),
                key_pad_mask,
            ], dim=1)
        else:
            pad = None

        out = self.transformer(seq, mask=mask, src_key_padding_mask=pad)
        # Drop the static prefix positions for the head.
        out = out[:, static_len:, :]  # (B, L, D)
        return self.head(out)  # (B, L, NUM_EVENT_TYPES)
