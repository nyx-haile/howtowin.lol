import torch
from model.heads import NextEventHead, OutcomeHead, NextDecisionHead, NextFrameHead
from model.tokens import NUM_EVENT_TYPES


def test_next_event_head_shape():
    h = NextEventHead(d_z=32, n_event_types=NUM_EVENT_TYPES)
    z = torch.randn(4, 32)
    logits = h(z)
    assert logits.shape == (4, NUM_EVENT_TYPES)


def test_outcome_head_returns_scalar_logit():
    h = OutcomeHead(d_z=32)
    z = torch.randn(4, 32)
    out = h(z)
    assert out.shape == (4,)


def test_next_decision_head_shape():
    n_decisions = 6
    n_participants = 10
    h = NextDecisionHead(d_z=32, n_decisions=n_decisions, n_participants=n_participants)
    z = torch.randn(4, 32)
    logits = h(z)
    assert logits.shape == (4, n_participants, n_decisions)


def test_next_frame_head_returns_mu_sigma():
    n_participants = 10
    feat_dim = 6
    h = NextFrameHead(d_z=32, n_participants=n_participants, feat_dim=feat_dim)
    z = torch.randn(4, 32)
    mu, logvar = h(z)
    assert mu.shape == (4, n_participants, feat_dim)
    assert logvar.shape == (4, n_participants, feat_dim)


def test_heads_outputs_are_finite():
    z = torch.randn(2, 32)
    assert torch.isfinite(NextEventHead(32, NUM_EVENT_TYPES)(z)).all()
    assert torch.isfinite(OutcomeHead(32)(z)).all()
    assert torch.isfinite(NextDecisionHead(32, 6, 10)(z)).all()
    mu, logvar = NextFrameHead(32, 10, 6)(z)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
