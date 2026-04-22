import torch

from model.heads import FactorizedNextEventHead, NextDecisionHead, NextFrameHead, OutcomeHead
from model.plan_b_model import D_R
from model.tokens import EVENT_TYPE_LABEL_COUNT, MAX_ITEM_ID, NUM_SLOTS, NUM_WARD_TYPES, SKILL_SLOT_COUNT


def test_factorized_next_event_head_shapes():
    h = FactorizedNextEventHead(d_in=D_R)
    r = torch.randn(4, D_R)
    out = h(r)
    assert out["type_logits"].shape == (4, EVENT_TYPE_LABEL_COUNT)
    assert out["actor_logits"].shape == (4, NUM_SLOTS)
    assert out["target_logits"].shape == (4, NUM_SLOTS)
    assert out["item_logits"].shape == (4, MAX_ITEM_ID + 1)
    assert out["skill_logits"].shape == (4, SKILL_SLOT_COUNT)
    assert out["ward_type_logits"].shape == (4, NUM_WARD_TYPES)


def test_outcome_head_returns_scalar_logit():
    h = OutcomeHead(d_in=D_R)
    r = torch.randn(4, D_R)
    out = h(r)
    assert out.shape == (4,)


def test_next_decision_head_shape():
    h = NextDecisionHead(d_in=D_R, n_decisions=6, n_participants=10)
    r = torch.randn(4, D_R)
    logits = h(r)
    assert logits.shape == (4, 10, 6)


def test_next_frame_head_returns_mu_sigma():
    h = NextFrameHead(d_in=D_R, n_participants=10, feat_dim=11)
    r = torch.randn(4, D_R)
    mu, logvar = h(r)
    assert mu.shape == (4, 10, 11)
    assert logvar.shape == (4, 10, 11)


def test_heads_outputs_are_finite():
    r = torch.randn(2, D_R)
    event = FactorizedNextEventHead(D_R)(r)
    for tensor in event.values():
        assert torch.isfinite(tensor).all()
    assert torch.isfinite(OutcomeHead(D_R)(r)).all()
    assert torch.isfinite(NextDecisionHead(D_R, 6, 10)(r)).all()
    mu, logvar = NextFrameHead(D_R, 10, 11)(r)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
