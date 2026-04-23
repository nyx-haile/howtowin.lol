import torch
import pytest
import torch.nn as nn
from model.rssm import RSSMCore, free_bits_kl


@pytest.fixture
def rssm():
    return RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)


def test_prior_shapes(rssm):
    h = torch.randn(2, 512)
    mu, logvar = rssm.prior(h)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)


def test_posterior_shapes(rssm):
    h = torch.randn(2, 512)
    o = torch.randn(2, 128)
    mu, logvar = rssm.posterior(h, o)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)


def test_step_returns_new_h_and_z(rssm):
    h = torch.randn(2, 512)
    z = torch.randn(2, 32)
    a = torch.randn(2, 256)
    h2 = rssm.step(h, z, a)
    assert h2.shape == (2, 512)


def test_free_bits_kl_non_negative():
    mu_q = torch.zeros(4, 32)
    logvar_q = torch.zeros(4, 32)
    mu_p = torch.zeros(4, 32)
    logvar_p = torch.zeros(4, 32)
    kl = free_bits_kl(mu_q, logvar_q, mu_p, logvar_p, free_bits_per_dim=0.5)
    assert torch.all(kl >= 0.0)


def test_free_bits_kl_clamps_below_floor():
    mu = torch.zeros(1, 32)
    logvar = torch.zeros(1, 32)
    kl = free_bits_kl(mu, logvar, mu, logvar, free_bits_per_dim=0.5)
    assert torch.allclose(kl, torch.tensor([16.0]))


def test_reparameterize_samples_with_gradient():
    from model.rssm import reparameterize
    mu = torch.zeros(2, 32, requires_grad=True)
    logvar = torch.zeros(2, 32, requires_grad=True)
    z = reparameterize(mu, logvar)
    z.sum().backward()
    assert mu.grad is not None
    assert logvar.grad is not None


def test_scan_preserves_hidden_dtype_when_gru_returns_lower_precision(rssm):
    class FakeGRU(nn.Module):
        def forward(self, packed, h0):
            return None, h0.to(torch.float16)

    rssm.gru = FakeGRU()
    h = torch.randn(2, 512, dtype=torch.float32)
    z = torch.randn(2, 32, dtype=torch.float32)
    action_sequence = torch.randn(2, 3, 256, dtype=torch.float32)
    lengths = torch.tensor([3, 1], dtype=torch.long)

    h_next = rssm.scan(h, z, action_sequence, lengths=lengths)

    assert h_next.dtype == torch.float32
