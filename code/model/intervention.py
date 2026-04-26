"""Step 4 / Gate D — counterfactual intervention surface.

Synthetic decision-token interventions on the rollout INPUT path. Per the
plan's revised Section 9, we do NOT perturb ``NextDecisionHead`` logits.
Instead we splice a synthetic decision token into the event window of a
target anchor and re-run :meth:`PlanBModel.rollout_prior_single`. Divergence
between the unperturbed and perturbed rollouts (Jensen-Shannon on the
deterministic prior win-probability sequence) is the candidate score.

The injection happens before calling ``rollout_prior_single`` — we extend
the flat ``token_embeddings`` table with one synthetic row and edit the
``event_window_positions / offsets / counts`` tensors so the new row is
the *first* event in the target window. ``rollout_prior_single`` itself is
untouched.
"""
from __future__ import annotations

from typing import Optional

import torch

from model.encoders import D_MODEL
from model.tokens import EVENT_TYPE_TO_ID, NUM_SLOTS

# Six player-controllable decision types. The plan calls these out
# explicitly; CHAMPION_KILL / BUILDING_KILL etc. are outcomes, not decisions.
INTERVENTION_DECISION_TYPES: tuple[str, ...] = (
    "ITEM_PURCHASED",
    "SKILL_LEVEL_UP",
    "WARD_PLACED",
    "RECALL",
    "ENGAGE",
    "DISENGAGE",
)

# Empirical thresholds. See ``intervention_driver.JS_DIVERGENCE_FLOOR /
# _SIGNAL_THRESHOLD`` for usage; exposed here for unit-test access.
JS_LOG2 = float(torch.tensor(2.0).log().item())  # ~0.6931, JS upper bound


@torch.no_grad()
def build_synthetic_token_embedding(
    model,
    *,
    decision_type: str,
    actor_slot: int = 1,
    timestamp_ms: int = 900_000,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Embed a single synthetic decision token; returns ``(D_MODEL,)``.

    Routes the synthetic token through the model's
    :class:`DynamicStreamEmbedder` so the result is calibrated to the same
    embedding scale that real events enter ``rollout_prior_single`` with.

    ``player_emb`` is zeroed — a synthetic token has no real player, and
    tying it to one would conflate per-player skill with the decision-type
    signal we're trying to isolate.
    """
    if decision_type not in INTERVENTION_DECISION_TYPES:
        raise ValueError(
            f"decision_type={decision_type!r} not in {INTERVENTION_DECISION_TYPES}"
        )
    if not (1 <= actor_slot < NUM_SLOTS):
        raise ValueError(f"actor_slot must be in 1..{NUM_SLOTS - 1}, got {actor_slot}")

    device = torch.device(device)
    type_id = EVENT_TYPE_TO_ID[decision_type]

    tokens = torch.tensor([[type_id]], dtype=torch.long, device=device)
    actors = torch.tensor([[actor_slot]], dtype=torch.long, device=device)
    timestamps = torch.tensor([[timestamp_ms]], dtype=torch.long, device=device)
    # gather over (B=1, NUM_SLOTS-1=10, D_MODEL); zeros so synthetic events
    # contribute no per-player signal.
    player_emb = torch.zeros(1, NUM_SLOTS - 1, D_MODEL, device=device)

    emb = model.dynamic_emb(tokens, actors, timestamps, player_emb)  # (1, 1, D_MODEL)
    return emb[0, 0]


def _recompute_offsets(counts: torch.Tensor) -> torch.Tensor:
    """Cumulative-sum offsets from a counts vector (matching collate_games)."""
    if counts.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=counts.device)
    z = torch.zeros(1, dtype=counts.dtype, device=counts.device)
    return torch.cat([z, counts[:-1].cumsum(0)]).to(torch.long)


def _inject_synthetic_token(
    *,
    token_embeddings: torch.Tensor,    # (L, D_MODEL) flat (already squeezed) OR (1, L, D_MODEL)
    event_window_positions: torch.Tensor,  # (total_events,) int64
    event_window_offsets: torch.Tensor,    # (T,) int64 — may have leading batch dim
    event_window_counts: torch.Tensor,     # (T,) int64 — may have leading batch dim
    inject_window_idx: int,
    synthetic_emb: torch.Tensor,           # (D_MODEL,)
    replace_window: bool = False,
    n_copies: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (token_embeddings_inj, positions_inj, offsets_inj, counts_inj).

    Two modes:

    - ``replace_window=False`` (additive): the synthetic embedding is appended
      as the new last row of ``token_embeddings`` and ``n_copies`` references
      to it are inserted at the **start** of window ``inject_window_idx``.
      Real events in that window are preserved.
    - ``replace_window=True`` (counterfactual): the target window's existing
      events are wiped and the window's ``count`` is set to ``n_copies``,
      pointing at the synthetic embedding. Real events outside that window
      are untouched. Use this when an additive perturbation is too weak — a
      well-trained RSSM is highly Lipschitz and a single extra token among
      many gets washed out.

    None of the input tensors is mutated.
    """
    if n_copies < 1:
        raise ValueError(f"n_copies must be >= 1, got {n_copies}")

    # Normalise shape: helper operates on flat (L, D) for the embedding table.
    if token_embeddings.dim() == 3:
        if token_embeddings.size(0) != 1:
            raise ValueError("token_embeddings batch dim must be 1 for intervention")
        token_emb_flat = token_embeddings[0]
    else:
        token_emb_flat = token_embeddings
    L = token_emb_flat.size(0)

    # Strip any leading batch dim on offsets / counts so we work with (T,).
    if event_window_offsets.dim() == 2:
        if event_window_offsets.size(0) != 1:
            raise ValueError("event_window_offsets batch dim must be 1")
        offsets = event_window_offsets[0]
    else:
        offsets = event_window_offsets
    if event_window_counts.dim() == 2:
        if event_window_counts.size(0) != 1:
            raise ValueError("event_window_counts batch dim must be 1")
        counts = event_window_counts[0]
    else:
        counts = event_window_counts

    T = counts.size(0)
    if not (0 <= inject_window_idx < T):
        raise ValueError(f"inject_window_idx={inject_window_idx} out of range [0, {T})")

    # 1. Append synthetic row to the flat token-embedding table.
    token_emb_inj = torch.cat(
        [token_emb_flat, synthetic_emb.unsqueeze(0).to(token_emb_flat.dtype)], dim=0,
    )
    new_position = L
    pos_dtype = event_window_positions.dtype
    pos_device = event_window_positions.device
    repeat_pos = torch.full((n_copies,), new_position, dtype=pos_dtype, device=pos_device)

    if replace_window:
        # Drop the target window's real positions; replace with n_copies refs to the synthetic.
        win_start = int(offsets[inject_window_idx].item())
        win_end = win_start + int(counts[inject_window_idx].item())
        positions_inj = torch.cat(
            [event_window_positions[:win_start], repeat_pos, event_window_positions[win_end:]],
            dim=0,
        )
        counts_inj = counts.clone()
        counts_inj[inject_window_idx] = n_copies
    else:
        # Additive: insert n_copies refs at the start of the target window.
        insert_at = int(offsets[inject_window_idx].item())
        positions_inj = torch.cat(
            [event_window_positions[:insert_at], repeat_pos, event_window_positions[insert_at:]],
            dim=0,
        )
        counts_inj = counts.clone()
        counts_inj[inject_window_idx] = counts_inj[inject_window_idx] + n_copies

    offsets_inj = _recompute_offsets(counts_inj.to(torch.long))
    return token_emb_inj, positions_inj, offsets_inj, counts_inj


@torch.no_grad()
def run_intervention_rollout(
    model,
    *,
    h0: torch.Tensor,
    z0: torch.Tensor,
    static_tokens: torch.Tensor,
    token_embeddings: torch.Tensor,
    event_window_positions: torch.Tensor,
    event_window_offsets: torch.Tensor,
    event_window_counts: torch.Tensor,
    anchor_mask: torch.Tensor,
    start_anchor: int,
    n_steps: int,
    n_valid_anchors: Optional[int],
    synthetic_emb: torch.Tensor,
    inject_at_step: int = 0,
    replace_window: bool = False,
    n_copies: int = 1,
    sustained: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Re-run :meth:`PlanBModel.rollout_prior_single` with a synthetic token.

    All tensors are forwarded verbatim to ``rollout_prior_single`` after the
    injection step; the function itself is not modified.

    - ``inject_at_step`` is relative to ``start_anchor`` — 0 means "inject in
      the first window the rollout consumes".
    - ``sustained=True`` injects in **every** rollout window (start_anchor
      through start_anchor+n_steps-1). Use this when single-window
      injection is too dilute — a well-trained RSSM relaxes back to its
      unperturbed trajectory within a few steps. Sustained injection holds
      the perturbation across the whole rollout horizon.
    - See :func:`_inject_synthetic_token` for ``replace_window`` and
      ``n_copies`` semantics.
    """
    tok = token_embeddings
    pos = event_window_positions
    off = event_window_offsets
    cnt = event_window_counts
    if sustained:
        n_valid = (
            int(anchor_mask.sum().item()) if n_valid_anchors is None else n_valid_anchors
        )
        max_step = min(n_steps, max(n_valid - start_anchor - 1, 0))
        steps = range(max_step)
    else:
        steps = (inject_at_step,)
    for s in steps:
        tok, pos, off, cnt = _inject_synthetic_token(
            token_embeddings=tok,
            event_window_positions=pos,
            event_window_offsets=off,
            event_window_counts=cnt,
            inject_window_idx=start_anchor + s,
            synthetic_emb=synthetic_emb,
            replace_window=replace_window,
            n_copies=n_copies,
        )
    tok_inj, pos_inj, off_inj, cnt_inj = tok, pos, off, cnt
    return model.rollout_prior_single(
        h0=h0,
        z0=z0,
        static_tokens=static_tokens,
        token_embeddings=tok_inj,
        event_window_positions=pos_inj,
        event_window_offsets=off_inj,
        event_window_counts=cnt_inj,
        anchor_mask=anchor_mask,
        start_anchor=start_anchor,
        n_steps=n_steps,
        n_valid_anchors=n_valid_anchors,
    )


def _outcome_probs(
    rollout: list[dict[str, torch.Tensor]],
    model,
) -> torch.Tensor:
    """Deterministic win-probabilities from a rollout output list.

    Uses ``[h || prior_mu]`` (the deterministic mean of the prior) rather
    than the sampled ``z`` to remove reparameterise() noise from the
    divergence comparison. Returns ``(n_steps,)`` float32 in [0, 1].

    Retained as a utility for downstream callers that want the outcome
    projection; the primary candidate divergence (see :func:`js_divergence`)
    is computed on the prior Gaussian itself, not on the outcome head, since
    a well-trained ``head_outcome`` saturates and is too flat to register
    single-token interventions.
    """
    if not rollout:
        return torch.zeros(0, dtype=torch.float32)
    reprs = []
    for step in rollout:
        h = step["h"]                # (1, D_H)
        mu = step["prior_mu"]        # (1, D_Z)
        reprs.append(torch.cat([h, mu], dim=-1))
    stacked = torch.cat(reprs, dim=0)            # (n_steps, D_R)
    logits = model.head_outcome(stacked)         # (n_steps,)
    return torch.sigmoid(logits).to(torch.float32)


def _bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(P || Q) for per-step Bernoullis. Numerically safe via clamping.

    eps must exceed float32 machine epsilon (~1.19e-7); otherwise ``1 - eps``
    rounds back to 1.0 and the clamp is a no-op for values at the boundary.
    """
    p = p.clamp(min=eps, max=1 - eps)
    q = q.clamp(min=eps, max=1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def _gaussian_symmetric_kl(
    mu_a: torch.Tensor, logvar_a: torch.Tensor,
    mu_b: torch.Tensor, logvar_b: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL between diagonal Gaussians N(μ_a, σ_a²) and N(μ_b, σ_b²).

    Returns a per-row scalar. KL(A||B) + KL(B||A), summed over the latent
    dim. Bounded below by 0; unbounded above. Sensitive to mean shifts.
    """
    var_a = logvar_a.exp()
    var_b = logvar_b.exp()
    diff_sq = (mu_a - mu_b).pow(2)
    kl_ab = 0.5 * (
        logvar_b - logvar_a + (var_a + diff_sq) / var_b - 1.0
    ).sum(dim=-1)
    kl_ba = 0.5 * (
        logvar_a - logvar_b + (var_b + diff_sq) / var_a - 1.0
    ).sum(dim=-1)
    return kl_ab + kl_ba


def js_divergence(
    base_outputs: list[dict[str, torch.Tensor]],
    perturbed_outputs: list[dict[str, torch.Tensor]],
    model,  # kept for signature compatibility; unused for prior-KL path
) -> float:
    """Mean per-step symmetric KL on the prior latent distributions.

    Despite the name (kept for API stability), this is **not** Jensen-Shannon
    on the outcome head — the trained ``head_outcome`` saturates and registers
    single-token interventions only at numerical floor. We instead score the
    intervention by how much the **prior Gaussian** N(prior_mu, σ²) differs
    between base and perturbed rollouts. That is the distribution the RSSM
    actually rolls forward; outcome / decision / event heads are linear
    readouts of it. Returns 0.0 for empty rollouts.
    """
    if not base_outputs or not perturbed_outputs:
        return 0.0
    n = min(len(base_outputs), len(perturbed_outputs))
    mus_a = torch.cat([s["prior_mu"] for s in base_outputs[:n]], dim=0)
    lvs_a = torch.cat([s["prior_logvar"] for s in base_outputs[:n]], dim=0)
    mus_b = torch.cat([s["prior_mu"] for s in perturbed_outputs[:n]], dim=0)
    lvs_b = torch.cat([s["prior_logvar"] for s in perturbed_outputs[:n]], dim=0)
    skl = _gaussian_symmetric_kl(mus_a, lvs_a, mus_b, lvs_b)
    return float(skl.mean().item())
