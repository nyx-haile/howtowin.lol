"""kNN retrieval over Plan B latent state.

Builds an index of [h_t || mu_q(z_t)] per training anchor, fits per-dim
whitening, and exposes batched cdist+topk queries. Backed by PyTorch
tensors only — no FAISS dependency. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import os
from dataclasses import dataclass
import torch

D_H = 512
D_Z = 32
KEY_DIM = D_H + D_Z  # 544

INDEX_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "retrieval"
)
DEFAULT_INDEX_PATH = os.path.join(INDEX_DIR, "plan_b_index.pt")

MID_GAME_MINUTES = tuple(range(10, 26))  # 10..25 inclusive
K_SWEEP = (16, 32, 64, 128, 256)
HEADLINE_K = 64
HEADLINE_GATE_BITS = 0.7

_SIGMA_FLOOR = 1e-6


class Whitener:
    """Per-dim z-score normalization. Params fit once on the corpus."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def fit(cls, corpus: torch.Tensor) -> "Whitener":
        mu = corpus.mean(dim=0)
        sigma = corpus.std(dim=0, unbiased=False).clamp(min=_SIGMA_FLOOR)
        return cls(mu=mu, sigma=sigma)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma

    def state_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "Whitener":
        return cls(mu=sd["mu"], sigma=sd["sigma"])


@torch.no_grad()
def encode_game_keys(model, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model.forward on a single-game batch (B=1) and return:

    keys:       (T, KEY_DIM) — concat of [h_t, post_mu_t] per anchor.
    minutes:    (T,) int64 — anchor minute = round(token_timestamp / 60000).
    blue_win:   () int8 — game-level outcome (1 if blue won, else 0).
    """
    assert batch["tokens"].size(0) == 1, "encode_game_keys expects B=1"
    was_training = model.training
    model.eval()
    out = model(batch)
    model.train(was_training)

    T = out["n_anchors"]
    h = out["h"][0]                # (T, D_H)
    post_mu = out["post_mu"][0]    # (T, D_Z)
    keys = torch.cat([h, post_mu], dim=-1)  # (T, KEY_DIM)

    anchor_pos = batch["anchor_positions"][0]   # (T,)
    ts = batch["token_timestamps"][0]           # (L,)
    anchor_ts = ts.gather(0, anchor_pos.long()) # (T,) ms
    minutes = (anchor_ts / 60000.0).round().to(torch.int64)

    blue_win = batch["outcome"][0].to(dtype=torch.int8)
    return keys, minutes, blue_win
