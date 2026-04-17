import math
import torch


def test_binary_entropy_at_half_is_one_bit():
    from model.m4_eval import binary_entropy
    p = torch.tensor([0.5])
    h = binary_entropy(p)
    assert torch.allclose(h, torch.tensor([1.0]), atol=1e-6)


def test_binary_entropy_at_endpoints_is_zero():
    from model.m4_eval import binary_entropy
    p = torch.tensor([0.0, 1.0])
    h = binary_entropy(p)
    assert torch.allclose(h, torch.tensor([0.0, 0.0]), atol=1e-6)


def test_binary_entropy_is_symmetric_around_half():
    from model.m4_eval import binary_entropy
    a = binary_entropy(torch.tensor([0.2]))
    b = binary_entropy(torch.tensor([0.8]))
    assert torch.allclose(a, b, atol=1e-6)


def test_binary_entropy_at_seventy_thirty_matches_formula():
    from model.m4_eval import binary_entropy
    p = 0.7
    expected = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    h = binary_entropy(torch.tensor([p]))
    assert abs(h.item() - expected) < 1e-6


def test_cohort_entropies_pure_cohort_has_zero_entropy():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3]])  # (1 query, k=4)
    blue_win = torch.tensor([1, 1, 1, 1], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert h.shape == (1,)
    assert abs(h[0].item() - 0.0) < 1e-6


def test_cohort_entropies_balanced_cohort_has_one_bit():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3]])
    blue_win = torch.tensor([1, 1, 0, 0], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert abs(h[0].item() - 1.0) < 1e-6


def test_cohort_entropies_per_query_independent():
    from model.m4_eval import cohort_entropies
    cohort_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    blue_win = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int8)
    h = cohort_entropies(cohort_idx, blue_win)
    assert h.shape == (2,)
    assert abs(h[0].item() - 0.0) < 1e-6  # all wins
    assert abs(h[1].item() - 1.0) < 1e-6  # half-half


def test_mean_entropy_at_k_averages():
    from model.m4_eval import mean_entropy_at_k
    cohort_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    blue_win = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int8)
    mean_h = mean_entropy_at_k(cohort_idx, blue_win)
    assert abs(mean_h - 0.5) < 1e-6
