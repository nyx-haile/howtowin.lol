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


def cohort_entropies(
    cohort_idx: torch.Tensor, blue_win: torch.Tensor
) -> torch.Tensor:
    """Per-query cohort outcome entropy in bits.

    cohort_idx: (Q, k) long — top-k corpus row indices per query.
    blue_win: (N,) int8 — per-corpus-row source-game outcome.
    Returns: (Q,) float entropies.
    """
    cohort_labels = blue_win[cohort_idx].float()    # (Q, k)
    p = cohort_labels.mean(dim=1)
    return binary_entropy(p)


def mean_entropy_at_k(
    cohort_idx: torch.Tensor, blue_win: torch.Tensor
) -> float:
    h = cohort_entropies(cohort_idx, blue_win)
    if h.numel() == 0:
        return float("nan")
    return float(h.mean().item())


def random_k_cohort_indices(*, Q: int, k: int, N: int, seed: int) -> torch.Tensor:
    """(Q, k) of corpus indices sampled uniformly without per-row replacement."""
    g = torch.Generator()
    g.manual_seed(seed)
    out = torch.empty(Q, k, dtype=torch.long)
    for q in range(Q):
        # without-replacement within a single cohort; with-replacement across queries.
        out[q] = torch.randperm(N, generator=g)[:k]
    return out


def per_minute_entropy_table(
    cohort_h: torch.Tensor,
    query_minutes: torch.Tensor,
    minutes,
) -> dict[int, float]:
    """Mean cohort entropy bucketed by query anchor minute."""
    out = {}
    for m in minutes:
        mask = (query_minutes == m)
        if mask.any():
            out[int(m)] = float(cohort_h[mask].mean().item())
        else:
            out[int(m)] = float("nan")
    return out
