import torch
import pytest


def test_whitener_fit_makes_corpus_unit_normal():
    from model.retrieval import Whitener
    torch.manual_seed(0)
    corpus = torch.randn(1024, 32) * 7.5 + 3.0
    w = Whitener.fit(corpus)
    out = w.apply(corpus)
    assert torch.allclose(out.mean(dim=0), torch.zeros(32), atol=1e-5)
    assert torch.allclose(out.std(dim=0, unbiased=False),
                          torch.ones(32), atol=1e-3)


def test_whitener_apply_uses_fitted_params_not_query_stats():
    from model.retrieval import Whitener
    # Deterministic corpus: mean=0, std=1, per dim.
    corpus = torch.tensor(
        [[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]]
    )
    w = Whitener.fit(corpus)
    # Query with values whose own mean is 100 — distinct from corpus mean 0.
    # If apply() wrongly used query-stats it would map these to zero.
    # With fitted params: (100 - 0) / 1 = 100.
    query = torch.full((3, 2), 100.0)
    out = w.apply(query)
    assert torch.allclose(out, torch.full((3, 2), 100.0), atol=1e-5)


def test_whitener_clamps_zero_std_dims():
    from model.retrieval import Whitener
    corpus = torch.zeros(100, 4)
    corpus[:, 0] = torch.arange(100, dtype=torch.float32)  # nonzero std
    # Dims 1, 2, 3 are constant → std==0; whitener must not produce NaN/Inf.
    w = Whitener.fit(corpus)
    out = w.apply(corpus)
    assert torch.isfinite(out).all()


def test_whitener_roundtrips_through_state_dict():
    from model.retrieval import Whitener
    torch.manual_seed(0)
    corpus = torch.randn(64, 16)
    w = Whitener.fit(corpus)
    sd = w.state_dict()
    w2 = Whitener.from_state_dict(sd)
    assert torch.equal(w.mu, w2.mu)
    assert torch.equal(w.sigma, w2.sigma)
