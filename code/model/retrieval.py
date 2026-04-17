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
