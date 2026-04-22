import torch

from model.heads import FactorizedNextEventHead, NextDecisionHead, NextFrameHead, OutcomeHead
from model.tokens import EVENT_TYPE_LABEL_COUNT, MAX_ITEM_ID, NUM_SLOTS, NUM_WARD_TYPES, SKILL_SLOT_COUNT


def test_factorized_next_event_head_shapes():
    h = FactorizedNextEventHead(d_in=32)
    z = torch.randn(4, 32)
    out = h(z)
    assert out["type_logits"].shape == (4, EVENT_TYPE_LABEL_COUNT)
    assert out["actor_logits"].shape == (4, NUM_SLOTS)
    assert out["target_logits"].shape == (4, NUM_SLOTS)
    assert out["item_logits"].shape == (4, MAX_ITEM_ID + 1)
    assert out["skill_logits"].shape == (4, SKILL_SLOT_COUNT)
    assert out["ward_type_logits"].shape == (4, NUM_WARD_TYPES)


def test_outcome_head_returns_scalar_logit():
    h = OutcomeHead(d_in=32)
    z = torch.randn(4, 32)
    out = h(z)
    assert out.shape == (4,)


def test_next_decision_head_shape():
    h = NextDecisionHead(d_in=32, n_decisions=6, n_participants=10)
    z = torch.randn(4, 32)
    logits = h(z)
    assert logits.shape == (4, 10, 6)


def test_next_frame_head_returns_mu_sigma():
    h = NextFrameHead(d_in=32, n_participants=10, feat_dim=11)
    z = torch.randn(4, 32)
    mu, logvar = h(z)
    assert mu.shape == (4, 10, 11)
    assert logvar.shape == (4, 10, 11)


def test_heads_outputs_are_finite():
    z = torch.randn(2, 32)
    event = FactorizedNextEventHead(32)(z)
    for tensor in event.values():
        assert torch.isfinite(tensor).all()
    assert torch.isfinite(OutcomeHead(32)(z)).all()
    assert torch.isfinite(NextDecisionHead(32, 6, 10)(z)).all()
    mu, logvar = NextFrameHead(32, 10, 11)(z)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
