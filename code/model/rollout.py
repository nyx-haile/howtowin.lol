import torch
from model.rssm import reparameterize


def rollout_prior(rssm, h0, z0, n_steps: int, action_summary=None):
    """Roll the prior forward n_steps without posterior updates.

    Returns a list of (h_t, z_t, mu_t, logvar_t) tuples of length n_steps.
    action_summary is reused at each step; pass None to use zeros.
    """
    if action_summary is None:
        action_summary = torch.zeros(h0.size(0), rssm.gru.input_size - rssm.d_z,
                                     device=h0.device)

    h = h0
    z = z0
    out = []
    for _ in range(n_steps):
        h = rssm.step(h, z, action_summary)
        mu, logvar = rssm.prior(h)
        z = reparameterize(mu, logvar)
        out.append((h, z, mu, logvar))
    return out
