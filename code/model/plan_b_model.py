"""Plan B composed model: encoders + RSSM + decoder heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dataset import FRAME_FEAT_DIM, MACRO_FEAT_DIM
from model.encoders import D_MODEL, DynamicStreamEmbedder, PlayerModelEncoder, StaticContextEncoder
from model.heads import FactorizedNextEventHead, NextDecisionHead, NextFrameHead, OutcomeHead
from model.rssm import RSSMCore, reparameterize
from model.static_features import STATIC_TOKEN_COUNT
from model.tokens import NUM_EVENT_TYPES

D_H = 512
D_Z = 32
D_OBS = 192
D_ACTION = D_MODEL
D_R = D_H + D_Z

N_PARTICIPANTS = 10
N_DECISION_TYPES = 6


class AnchorObservationProjector(nn.Module):
    """Projects per-anchor participant + macro observations into D_OBS."""

    def __init__(self, d_obs: int = D_OBS):
        super().__init__()
        input_dim = N_PARTICIPANTS * FRAME_FEAT_DIM + MACRO_FEAT_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, d_obs),
        )

    def forward(self, frame_feats: torch.Tensor, macro_feats: torch.Tensor) -> torch.Tensor:
        b, t, p, f = frame_feats.shape
        flat = frame_feats.view(b, t, p * f)
        fused = torch.cat([flat, macro_feats], dim=-1)
        return self.net(fused)


class PlanBModel(nn.Module):
    def __init__(self, max_puuids: int):
        super().__init__()
        self.static_enc = StaticContextEncoder()
        self.player_enc = PlayerModelEncoder(max_puuids=max_puuids)
        self.dynamic_emb = DynamicStreamEmbedder()

        self.obs_proj = AnchorObservationProjector(d_obs=D_OBS)

        self.h0 = nn.Parameter(torch.zeros(D_H))
        self.z0 = nn.Parameter(torch.zeros(D_Z))
        self.static_num_heads = 4
        self.static_head_dim = D_MODEL // self.static_num_heads
        if self.static_head_dim * self.static_num_heads != D_MODEL:
            raise ValueError("D_MODEL must be divisible by static_num_heads")
        self.static_attn_scale = self.static_head_dim ** -0.5
        self.static_anchor_query = nn.Linear(D_H + D_Z, D_MODEL)
        self.static_event_query = nn.Linear(D_ACTION + D_Z, D_MODEL)
        self.static_key = nn.Linear(D_MODEL, D_MODEL)
        self.static_value = nn.Linear(D_MODEL, D_MODEL)
        self.static_attn_out = nn.Linear(D_MODEL, D_MODEL)
        self.static_to_action = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_ACTION),
        )
        self.static_to_obs = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_OBS),
        )
        self.static_summary = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_H),
        )
        self.rssm = RSSMCore(d_h=D_H, d_z=D_Z, d_action=D_ACTION, d_obs=D_OBS)

        self.head_event = FactorizedNextEventHead(D_R)
        self.head_outcome = OutcomeHead(D_R)
        self.head_decision = NextDecisionHead(D_R, N_DECISION_TYPES, N_PARTICIPANTS)
        self.head_frame = NextFrameHead(D_R, N_PARTICIPANTS, FRAME_FEAT_DIM)

    def _reshape_event_outputs(self, event_out: dict[str, torch.Tensor], B: int, T: int):
        reshaped = {}
        for key, value in event_out.items():
            reshaped[key] = value.view(B, T, -1)
        return reshaped

    def encode_static_tokens(self, static_batch: torch.Tensor) -> torch.Tensor:
        return self.static_enc(static_batch)

    def encode_static_summary(self, static_batch: torch.Tensor) -> torch.Tensor:
        static_tokens = self.encode_static_tokens(static_batch)
        return self.static_summary(static_tokens.mean(dim=1))

    def embed_token_sequence(self, batch, player_emb: torch.Tensor | None = None) -> torch.Tensor:
        if player_emb is None:
            player_emb = self.player_enc(batch["players"], batch["player_ids"])
        return self.dynamic_emb(
            batch["tokens"],
            batch["token_actors"],
            batch["token_timestamps"],
            player_emb,
            targets=batch.get("token_targets"),
            item_ids=batch.get("token_item_ids"),
            skill_slots=batch.get("token_skill_slots"),
            monster_types=batch.get("token_monster_types"),
            monster_subtypes=batch.get("token_monster_subtypes"),
            building_types=batch.get("token_building_types"),
            lane_types=batch.get("token_lane_types"),
            tower_types=batch.get("token_tower_types"),
            ward_types=batch.get("token_ward_types"),
        )

    def anchor_representation(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)

    def decode_anchor_repr(self, anchor_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        event_out = self.head_event(anchor_repr)
        outcome_logits = self.head_outcome(anchor_repr)
        decision_logits = self.head_decision(anchor_repr)
        frame_mu, frame_lv = self.head_frame(anchor_repr)
        return {
            "event_factors": event_out,
            "outcome_logits": outcome_logits,
            "decision_logits": decision_logits,
            "frame_mu": frame_mu,
            "frame_logvar": frame_lv,
        }

    def _zero_action(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, D_ACTION, device=device)

    def _prepare_static_attention(self, static_tokens: torch.Tensor):
        bsz, n_tokens, _ = static_tokens.shape
        key = self.static_key(static_tokens).view(
            bsz, n_tokens, self.static_num_heads, self.static_head_dim
        ).transpose(1, 2)
        value = self.static_value(static_tokens).view(
            bsz, n_tokens, self.static_num_heads, self.static_head_dim
        ).transpose(1, 2)
        return key, value

    def _static_context_from_query(
        self,
        query: torch.Tensor,
        static_key: torch.Tensor,
        static_value: torch.Tensor,
        *,
        need_weights: bool = False,
    ):
        squeeze_steps = query.dim() == 2
        if squeeze_steps:
            query = query.unsqueeze(1)

        bsz, n_steps, _ = query.shape
        query = query.view(
            bsz, n_steps, self.static_num_heads, self.static_head_dim
        ).transpose(1, 2)

        if need_weights:
            attn_scores = torch.matmul(query, static_key.transpose(-2, -1)) * self.static_attn_scale
            attn_weights = attn_scores.softmax(dim=-1)
            context = torch.matmul(attn_weights, static_value)
            mean_weights = attn_weights.mean(dim=1)
        else:
            context = F.scaled_dot_product_attention(query, static_key, static_value, dropout_p=0.0)
            mean_weights = None

        context = context.transpose(1, 2).contiguous().view(bsz, n_steps, D_MODEL)
        context = self.static_attn_out(context)

        if squeeze_steps:
            context = context.squeeze(1)
            if mean_weights is not None:
                mean_weights = mean_weights.squeeze(1)
        return context, mean_weights

    def _static_context(self, h, z, action_input, static_key, static_value, need_weights: bool = False):
        del action_input
        query = self.static_anchor_query(torch.cat([h, z], dim=-1))
        return self._static_context_from_query(
            query,
            static_key,
            static_value,
            need_weights=need_weights,
        )

    def _static_event_context(self, z, event_window_emb, static_key, static_value):
        if event_window_emb is None:
            return None
        z_query = z.unsqueeze(1).expand(-1, event_window_emb.size(1), -1)
        query = self.static_event_query(torch.cat([event_window_emb, z_query], dim=-1))
        context, _ = self._static_context_from_query(
            query,
            static_key,
            static_value,
            need_weights=False,
        )
        return context

    def _gather_event_window(self, token_emb: torch.Tensor, batch, anchor_idx: int):
        offsets = batch["event_window_offsets"][:, anchor_idx]
        counts = batch["event_window_counts"][:, anchor_idx]
        precomputed = batch.get("max_event_window_per_anchor")
        if precomputed is not None:
            max_count = precomputed[anchor_idx] if counts.numel() else 0
        else:
            max_count = int(counts.max().item()) if counts.numel() else 0
        if max_count == 0:
            return None, None

        flat_positions = batch["event_window_positions"]
        if flat_positions.numel() == 0:
            return None, None

        B, _L, D = token_emb.shape
        device = token_emb.device
        step_offsets = torch.arange(max_count, device=device).unsqueeze(0)
        mask = step_offsets < counts.unsqueeze(1)
        flat_idx = offsets.unsqueeze(1) + step_offsets
        safe_flat_idx = torch.where(mask, flat_idx, torch.zeros_like(flat_idx))
        positions = flat_positions.index_select(0, safe_flat_idx.reshape(-1)).view(B, max_count)
        gather_idx = positions.unsqueeze(-1).expand(-1, -1, D)
        events = torch.gather(token_emb, 1, gather_idx)
        return events, mask

    def _advance_event_window(self, h, z, event_window_emb, event_mask, static_key, static_value):
        if event_window_emb is None or event_mask is None:
            return h
        lengths = event_mask.to(dtype=torch.long).sum(dim=1)

        static_ctx = self._static_event_context(z, event_window_emb, static_key, static_value)
        conditioned_action = event_window_emb + self.static_to_action(static_ctx)
        return self.rssm.scan(
            h,
            z,
            conditioned_action,
            lengths=lengths,
        )

    def rollout_prior_single(
        self,
        *,
        h0: torch.Tensor,
        z0: torch.Tensor,
        static_tokens: torch.Tensor,
        token_embeddings: torch.Tensor,
        event_window_positions: torch.Tensor,
        event_window_offsets: torch.Tensor,
        event_window_counts: torch.Tensor,
        anchor_mask: torch.Tensor,
        start_anchor: int,
        n_steps: int,
        n_valid_anchors: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        if h0.dim() == 1:
            h = h0.unsqueeze(0)
        else:
            h = h0
        if z0.dim() == 1:
            z = z0.unsqueeze(0)
        else:
            z = z0
        if static_tokens.dim() == 2:
            static_tokens = static_tokens.unsqueeze(0)
        if token_embeddings.dim() == 3:
            token_embeddings = token_embeddings[0]
        static_key, static_value = self._prepare_static_attention(static_tokens)

        zero_action = self._zero_action(h.size(0), h.device)
        outputs = []
        if n_valid_anchors is None:
            n_valid_anchors = int(anchor_mask.sum().item())
        max_steps = min(n_steps, max(n_valid_anchors - start_anchor - 1, 0))

        gpu_device = h.device
        for step_idx in range(max_steps):
            window_idx = start_anchor + step_idx
            count = int(event_window_counts[window_idx].item())
            if count > 0:
                start = int(event_window_offsets[window_idx].item())
                positions = event_window_positions[start:start + count].to(gpu_device)
                window_events = token_embeddings.index_select(0, positions).unsqueeze(0)
                window_mask = torch.ones(1, count, dtype=torch.bool, device=gpu_device)
                h = self._advance_event_window(h, z, window_events, window_mask, static_key, static_value)

            static_ctx, _ = self._static_context(
                h, z, zero_action, static_key, static_value, need_weights=False
            )
            prior_mu, prior_logvar = self.rssm.prior(h)
            z = reparameterize(prior_mu, prior_logvar)
            anchor_repr = self.anchor_representation(h, z)
            outputs.append(
                {
                    "h": h,
                    "z": z,
                    "prior_mu": prior_mu,
                    "prior_logvar": prior_logvar,
                    "anchor_repr": anchor_repr,
                    "static_context": static_ctx,
                }
            )
        return outputs

    def forward(self, batch):
        B = batch["tokens"].size(0)
        T = batch["anchor_positions"].size(1)

        static_tokens = self.encode_static_tokens(batch["static"])
        if static_tokens.size(1) != STATIC_TOKEN_COUNT:
            raise RuntimeError(f"expected {STATIC_TOKEN_COUNT} static tokens, got {static_tokens.size(1)}")
        static_key, static_value = self._prepare_static_attention(static_tokens)
        player_emb = self.player_enc(batch["players"], batch["player_ids"])
        token_emb = self.embed_token_sequence(batch, player_emb=player_emb)

        obs = self.obs_proj(batch["frame_features"], batch["anchor_macro_features"])
        anchor_mask = batch["anchor_mask"]
        zero_action = self._zero_action(B, token_emb.device)

        h = self.h0.unsqueeze(0).expand(B, -1)
        z = self.z0.unsqueeze(0).expand(B, -1)

        post_mu_all, post_logvar_all = [], []
        prior_mu_all, prior_logvar_all = [], []
        z_all, h_all = [], []
        static_ctx_all = []

        for t in range(T):
            valid_anchor = anchor_mask[:, t].unsqueeze(-1)
            mask = valid_anchor.to(h.dtype)
            static_ctx, _ = self._static_context(
                h, z, zero_action, static_key, static_value, need_weights=False
            )
            conditioned_obs = obs[:, t] + self.static_to_obs(static_ctx)
            pr_mu, pr_lv = self.rssm.prior(h)
            po_mu, po_lv = self.rssm.posterior(h, conditioned_obs)
            z_post = reparameterize(po_mu, po_lv)
            z = torch.where(valid_anchor, z_post, z)

            h_all.append(h * mask)
            z_all.append(z * mask)
            prior_mu_all.append(pr_mu * mask)
            prior_logvar_all.append(pr_lv * mask)
            post_mu_all.append(po_mu * mask)
            post_logvar_all.append(po_lv * mask)
            static_ctx_all.append(static_ctx * mask)

            if t < T - 1:
                event_window_emb, event_mask = self._gather_event_window(token_emb, batch, t)
                h = self._advance_event_window(h, z, event_window_emb, event_mask, static_key, static_value)

        Z = torch.stack(z_all, dim=1)
        H = torch.stack(h_all, dim=1)
        post_mu = torch.stack(post_mu_all, dim=1)
        post_lv = torch.stack(post_logvar_all, dim=1)
        prior_mu = torch.stack(prior_mu_all, dim=1)
        prior_lv = torch.stack(prior_logvar_all, dim=1)
        static_context = torch.stack(static_ctx_all, dim=1)
        anchor_repr = self.anchor_representation(H, Z)

        decoded = self.decode_anchor_repr(anchor_repr.view(B * T, D_R))
        event_out = self._reshape_event_outputs(decoded["event_factors"], B, T)
        event_logits = event_out["type_logits"][..., 1:1 + NUM_EVENT_TYPES]
        outcome_logits = decoded["outcome_logits"].view(B, T)
        decision_logits = decoded["decision_logits"].view(B, T, N_PARTICIPANTS, N_DECISION_TYPES)
        frame_mu = decoded["frame_mu"].view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)
        frame_lv = decoded["frame_logvar"].view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)

        return {
            "event_logits": event_logits,
            "event_factors": event_out,
            "outcome_logits": outcome_logits,
            "decision_logits": decision_logits,
            "frame_mu": frame_mu,
            "frame_logvar": frame_lv,
            "post_mu": post_mu,
            "post_logvar": post_lv,
            "prior_mu": prior_mu,
            "prior_logvar": prior_lv,
            "z": Z,
            "h": H,
            "anchor_repr": anchor_repr,
            "anchor_mask": anchor_mask,
            "static_tokens": static_tokens,
            "static_context": static_context,
            "token_embeddings": token_emb,
            "n_anchors": T,
            "h_final": h,
            "z_final": z,
        }
