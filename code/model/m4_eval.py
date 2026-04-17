"""M4 retrieval-check eval harness.

Builds queries from holdout splits, runs k-sweep entropy, prints
per-minute table, runs baseline indexes, writes report. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import math
import torch

_LOG2 = math.log(2.0)


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-element binary entropy in bits. Returns 0 at p in {0, 1}."""
    # torch.xlogy(x, y) = x * log(y), with xlogy(0, 0) = 0 by convention,
    # so this is numerically correct at both endpoints without clamping.
    h_nats = -(torch.xlogy(p, p) + torch.xlogy(1.0 - p, 1.0 - p))
    return h_nats / _LOG2
