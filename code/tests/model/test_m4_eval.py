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


def test_random_k_cohort_indices_shape_and_range():
    from model.m4_eval import random_k_cohort_indices
    torch.manual_seed(0)
    idx = random_k_cohort_indices(Q=10, k=4, N=100, seed=0)
    assert idx.shape == (10, 4)
    assert (idx >= 0).all() and (idx < 100).all()


def test_random_k_cohort_indices_deterministic_under_seed():
    from model.m4_eval import random_k_cohort_indices
    a = random_k_cohort_indices(Q=10, k=4, N=100, seed=42)
    b = random_k_cohort_indices(Q=10, k=4, N=100, seed=42)
    assert torch.equal(a, b)


def test_per_minute_entropy_table_groups_by_query_minute():
    from model.m4_eval import per_minute_entropy_table
    cohort_h = torch.tensor([0.5, 0.7, 0.9, 1.0])
    query_minutes = torch.tensor([10, 10, 15, 20])
    table = per_minute_entropy_table(cohort_h, query_minutes,
                                     minutes=(10, 15, 20))
    # float32 mean → Python float: cannot use exact equality.
    assert abs(table[10] - (0.5 + 0.7) / 2) < 1e-6
    assert abs(table[15] - 0.9) < 1e-6
    assert abs(table[20] - 1.0) < 1e-6
