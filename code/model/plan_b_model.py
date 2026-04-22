"""Plan B composed model: encoders + RSSM + decoder heads."""
from __future__ import annotations

import torch
import torch.nn as nn

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


class ActionSummarizer(nn.Module):
    """Pools event-window token embeddings into a per-anchor action summary."""

    def __init__(self, d_action: int = D_ACTION, d_model: int = D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d_model, d_action)

    def forward(self, event_window_embeddings, window_mask):
        mask = window_mask.unsqueeze(-1).float()
        summed = (event_window_embeddings * mask).sum(dim=2)
        counts = mask.sum(dim=2).clamp(min=1.0)
        mean = summed / counts
        return self.proj(mean)


class PlanBModel(nn.Module):
    def __init__(self, max_puuids: int):
        super().__init__()
        self.static_enc = StaticContextEncoder()
        self.player_enc = PlayerModelEncoder(max_puuids=max_puuids)
        self.dynamic_emb = DynamicStreamEmbedder()

        self.obs_proj = AnchorObservationProjector(d_obs=D_OBS)
        self.action_summarizer = ActionSummarizer(d_action=D_ACTION, d_model=D_MODEL)

        self.h0 = nn.Parameter(torch.zeros(D_H))
        self.z0 = nn.Parameter(torch.zeros(D_Z))
        self.static_step_query = nn.Linear(D_H + D_Z + D_ACTION, D_MODEL)
        self.static_cross_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=4,
            batch_first=True,
        )
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

        self.head_event = FactorizedNextEventHead(D_Z)
        self.head_outcome = OutcomeHead(D_Z)
        self.head_decision = NextDecisionHead(D_Z, N_DECISION_TYPES, N_PARTICIPANTS)
        self.head_frame = NextFrameHead(D_Z, N_PARTICIPANTS, FRAME_FEAT_DIM)

    def _gather_event_windows(self, token_emb, batch):
        raw = batch["event_window_embeddings_raw"]
        B, T, W = raw.shape
        D = token_emb.size(-1)
        idx = raw.view(B, T * W).unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(token_emb, 1, idx).view(B, T, W, D)
        return gathered

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

    def _static_context(self, h, z, action_summary, static_tokens):
        query = self.static_step_query(torch.cat([h, z, action_summary], dim=-1)).unsqueeze(1)
        context, attn_weights = self.static_cross_attn(query, static_tokens, static_tokens, need_weights=True)
        return context.squeeze(1), attn_weights.squeeze(1)

    def forward(self, batch):
        B = batch["tokens"].size(0)

        static_tokens = self.encode_static_tokens(batch["static"])
        if static_tokens.size(1) != STATIC_TOKEN_COUNT:
            raise RuntimeError(f"expected {STATIC_TOKEN_COUNT} static tokens, got {static_tokens.size(1)}")
        player_emb = self.player_enc(batch["players"], batch["player_ids"])
        token_emb = self.dynamic_emb(
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

        frame_features = batch["frame_features"]
        macro_features = batch["anchor_macro_features"]
        window_mask = batch["window_mask"]
        event_window_emb = self._gather_event_windows(token_emb, batch)

        T = batch["anchor_positions"].size(1)

        obs = self.obs_proj(frame_features, macro_features)
        action_summary = self.action_summarizer(event_window_emb, window_mask)

        h = self.h0.unsqueeze(0).expand(B, -1)
        z = self.z0.unsqueeze(0).expand(B, -1)

        post_mu_all, post_logvar_all = [], []
        prior_mu_all, prior_logvar_all = [], []
        z_all, h_all = [], []
        static_ctx_all, static_attn_all = [], []

        for t in range(T):
            static_ctx, static_attn = self._static_context(h, z, action_summary[:, t], static_tokens)
            conditioned_action = action_summary[:, t] + self.static_to_action(static_ctx)
            conditioned_obs = obs[:, t] + self.static_to_obs(static_ctx)
            h = self.rssm.step(h, z, conditioned_action)
            h_all.append(h)
            pr_mu, pr_lv = self.rssm.prior(h)
            po_mu, po_lv = self.rssm.posterior(h, conditioned_obs)
            z = reparameterize(po_mu, po_lv)
            post_mu_all.append(po_mu)
            post_logvar_all.append(po_lv)
            prior_mu_all.append(pr_mu)
            prior_logvar_all.append(pr_lv)
            z_all.append(z)
            static_ctx_all.append(static_ctx)
            static_attn_all.append(static_attn)

        Z = torch.stack(z_all, dim=1)
        H = torch.stack(h_all, dim=1)
        post_mu = torch.stack(post_mu_all, dim=1)
        post_lv = torch.stack(post_logvar_all, dim=1)
        prior_mu = torch.stack(prior_mu_all, dim=1)
        prior_lv = torch.stack(prior_logvar_all, dim=1)
        static_context = torch.stack(static_ctx_all, dim=1)
        static_attn = torch.stack(static_attn_all, dim=1)

        z_flat = Z.view(B * T, D_Z)
        event_out = self.head_event(z_flat)
        event_out = self._reshape_event_outputs(event_out, B, T)
        event_logits = event_out["type_logits"][..., 1:1 + NUM_EVENT_TYPES]
        outcome_logits = self.head_outcome(z_flat).view(B, T)
        decision_logits = self.head_decision(z_flat).view(B, T, N_PARTICIPANTS, N_DECISION_TYPES)
        frame_mu, frame_lv = self.head_frame(z_flat)
        frame_mu = frame_mu.view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)
        frame_lv = frame_lv.view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)

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
            "static_tokens": static_tokens,
            "static_context": static_context,
            "static_attention_weights": static_attn,
            "n_anchors": T,
            "h_final": h,
            "z_final": z,
        }
