import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """KL(q || p) per-sample, summed across last dim. Shape: (batch,)."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
        + logvar_p - logvar_q
    )
    return kl.sum(dim=-1)


def free_bits_kl(mu_q, logvar_q, mu_p, logvar_p, free_bits_per_dim: float) -> torch.Tensor:
    """Free-bits KL. Clamps per-dim KL to a floor, then sums. Shape: (batch,)."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    per_dim = 0.5 * (
        (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
        + logvar_p - logvar_q
    )
    floored = per_dim.clamp(min=free_bits_per_dim)
    return floored.sum(dim=-1)


class RSSMCore(nn.Module):
    """Recurrent state-space model core: GRU deterministic path + stochastic latent."""

    def __init__(self, d_h: int = 512, d_z: int = 32, d_action: int = 256,
                 d_obs: int = 128, d_hidden: int = 512):
        super().__init__()
        self.d_h = d_h
        self.d_z = d_z

        self.gru = nn.GRUCell(input_size=d_action + d_z, hidden_size=d_h)

        self.prior_net = nn.Sequential(
            nn.Linear(d_h, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * d_z),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(d_h + d_obs, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * d_z),
        )

    def prior(self, h):
        out = self.prior_net(h)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-10.0, max=10.0)

    def posterior(self, h, o):
        out = self.posterior_net(torch.cat([h, o], dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-10.0, max=10.0)

    def step(self, h, z, action_summary):
        """Advance h given previous z and the action/event summary for the window."""
        inp = torch.cat([action_summary, z], dim=-1)
        return self.gru(inp, h)
