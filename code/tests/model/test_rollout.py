import torch
from model.rssm import RSSMCore
from model.rollout import rollout_prior


def test_rollout_returns_correct_number_of_steps():
    rssm = RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)
    h0 = torch.randn(2, 512)
    z0 = torch.randn(2, 32)
    steps = rollout_prior(rssm, h0, z0, n_steps=3, action_summary=torch.randn(2, 256))
    assert len(steps) == 3
    for h, z, mu, logvar in steps:
        assert h.shape == (2, 512)
        assert z.shape == (2, 32)
        assert mu.shape == (2, 32)
        assert logvar.shape == (2, 32)


def test_rollout_without_action_uses_zeros():
    rssm = RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)
    h0 = torch.zeros(1, 512)
    z0 = torch.zeros(1, 32)
    steps = rollout_prior(rssm, h0, z0, n_steps=2, action_summary=None)
    assert len(steps) == 2


def test_rollout_gradient_flows_to_prior_net():
    rssm = RSSMCore(d_h=512, d_z=32, d_action=256, d_obs=128)
    h0 = torch.randn(1, 512)
    z0 = torch.randn(1, 32)
    steps = rollout_prior(rssm, h0, z0, n_steps=3,
                          action_summary=torch.randn(1, 256))
    loss = sum(z.sum() for _h, z, _mu, _lv in steps)
    loss.backward()
    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in rssm.prior_net.parameters())
    assert any_grad
