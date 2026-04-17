"""Plan B composed model: encoders + RSSM + decoder heads.

Iterates per-anchor over the RSSM, using the posterior during training
and the prior during imagination rollouts. Four heads decode z_t at
each anchor step.
"""
import torch
import torch.nn as nn
from model.encoders import (
    StaticContextEncoder, PlayerModelEncoder, DynamicStreamEmbedder, D_MODEL,
)
from model.patch_params import PATCH_VECTOR_DIM
from model.rssm import RSSMCore, reparameterize
from model.heads import NextEventHead, OutcomeHead, NextDecisionHead, NextFrameHead
from model.tokens import NUM_EVENT_TYPES, ANCHOR_TOKEN

D_H = 512
D_Z = 32
D_OBS = 128
D_ACTION = D_MODEL  # action summary dim matches dynamic embedder output

N_PARTICIPANTS = 10
N_DECISION_TYPES = 6
FRAME_FEAT_DIM = 6


class AnchorObservationProjector(nn.Module):
    """Projects per-anchor per-participant frame features into D_OBS."""
    def __init__(self, d_obs: int = D_OBS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_PARTICIPANTS * FRAME_FEAT_DIM, 256), nn.GELU(),
            nn.Linear(256, d_obs),
        )

    def forward(self, frame_feats):
        # frame_feats: (B, T, 10, 6)
        b, t, p, f = frame_feats.shape
        flat = frame_feats.view(b, t, p * f)
        return self.net(flat)


class ActionSummarizer(nn.Module):
    """Pools event-window token embeddings into a per-anchor action summary."""
    def __init__(self, d_action: int = D_ACTION, d_model: int = D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d_model, d_action)

    def forward(self, event_window_embeddings, window_mask):
        # event_window_embeddings: (B, T, W, D_MODEL), window_mask: (B, T, W)
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

        # Static seeds the recurrence.
        self.static_to_h = nn.Linear(D_MODEL, D_H)

        self.rssm = RSSMCore(d_h=D_H, d_z=D_Z, d_action=D_ACTION, d_obs=D_OBS)

        self.head_event = NextEventHead(D_Z, NUM_EVENT_TYPES)
        self.head_outcome = OutcomeHead(D_Z)
        self.head_decision = NextDecisionHead(D_Z, N_DECISION_TYPES, N_PARTICIPANTS)
        self.head_frame = NextFrameHead(D_Z, N_PARTICIPANTS, FRAME_FEAT_DIM)

    def _gather_event_windows(self, token_emb, batch):
        """Gather token embeddings at event-window positions."""
        raw = batch["event_window_embeddings_raw"]  # (B, T, W) indices into token seq
        B, T, W = raw.shape
        D = token_emb.size(-1)
        idx = raw.view(B, T * W).unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(token_emb, 1, idx).view(B, T, W, D)
        return gathered

    def forward(self, batch):
        B = batch["tokens"].size(0)

        static = self.static_enc(batch["static"])  # (B, D_MODEL)
        player_emb = self.player_enc(batch["players"], batch["player_ids"])  # (B, 10, D_MODEL)
        token_emb = self.dynamic_emb(
            batch["tokens"], batch["token_actors"],
            batch["token_timestamps"], player_emb,
        )  # (B, L, D_MODEL)

        # Per-anchor tensors from dataset/collate.
        frame_features = batch["frame_features"]   # (B, T, 10, 6)
        window_mask = batch["window_mask"]          # (B, T, W)
        event_window_emb = self._gather_event_windows(token_emb, batch)  # (B, T, W, D_MODEL)

        T = batch["anchor_positions"].size(1)

        obs = self.obs_proj(frame_features)                    # (B, T, D_OBS)
        action_summary = self.action_summarizer(event_window_emb, window_mask)  # (B, T, D_ACTION)

        h = self.static_to_h(static)  # (B, D_H)
        z = torch.zeros(B, D_Z, device=h.device)

        post_mu_all, post_logvar_all = [], []
        prior_mu_all, prior_logvar_all = [], []
        z_all, h_all = [], []

        for t in range(T):
            h = self.rssm.step(h, z, action_summary[:, t])
            h_all.append(h)
            pr_mu, pr_lv = self.rssm.prior(h)
            po_mu, po_lv = self.rssm.posterior(h, obs[:, t])
            z = reparameterize(po_mu, po_lv)
            post_mu_all.append(po_mu)
            post_logvar_all.append(po_lv)
            prior_mu_all.append(pr_mu)
            prior_logvar_all.append(pr_lv)
            z_all.append(z)

        Z = torch.stack(z_all, dim=1)                # (B, T, D_Z)
        H = torch.stack(h_all, dim=1)                 # (B, T, D_H)
        post_mu = torch.stack(post_mu_all, dim=1)
        post_lv = torch.stack(post_logvar_all, dim=1)
        prior_mu = torch.stack(prior_mu_all, dim=1)
        prior_lv = torch.stack(prior_logvar_all, dim=1)

        # Heads over flattened Z.
        z_flat = Z.view(B * T, D_Z)
        event_logits = self.head_event(z_flat).view(B, T, NUM_EVENT_TYPES)
        outcome_logits = self.head_outcome(z_flat).view(B, T)
        decision_logits = self.head_decision(z_flat).view(B, T, N_PARTICIPANTS, N_DECISION_TYPES)
        frame_mu, frame_lv = self.head_frame(z_flat)
        frame_mu = frame_mu.view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)
        frame_lv = frame_lv.view(B, T, N_PARTICIPANTS, FRAME_FEAT_DIM)

        return {
            "event_logits": event_logits,
            "outcome_logits": outcome_logits,
            "decision_logits": decision_logits,
            "frame_mu": frame_mu,
            "frame_logvar": frame_lv,
            "post_mu": post_mu, "post_logvar": post_lv,
            "prior_mu": prior_mu, "prior_logvar": prior_lv,
            "z": Z,
            "h": H,
            "n_anchors": T,
            "h_final": h, "z_final": z,
        }
