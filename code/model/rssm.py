import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


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

        self.gru = nn.GRU(input_size=d_action + d_z, hidden_size=d_h, batch_first=True)

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
        inp = torch.cat([action_summary, z], dim=-1).unsqueeze(1)
        _, h_next = self.gru(inp, h.unsqueeze(0))
        return h_next.squeeze(0)

    def scan(self, h, z, action_sequence, mask: torch.Tensor | None = None,
             lengths: torch.Tensor | None = None):
        """Advance h over an ordered action/event sequence, ignoring padded suffixes."""
        if action_sequence is None:
            return h
        if action_sequence.dim() != 3:
            raise ValueError("action_sequence must have shape (batch, steps, d_action)")

        batch_size, max_steps, _ = action_sequence.shape
        if max_steps == 0:
            return h

        if lengths is None:
            if mask is None:
                lengths = torch.full(
                    (batch_size,),
                    max_steps,
                    dtype=torch.long,
                    device=action_sequence.device,
                )
            else:
                lengths = mask.to(dtype=torch.long).sum(dim=1)
        else:
            lengths = lengths.to(device=action_sequence.device, dtype=torch.long)

        active_idx = torch.nonzero(lengths > 0, as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            return h

        active_lengths = lengths.index_select(0, active_idx)
        active_max_steps = int(active_lengths.max().item())
        if active_max_steps == 0:
            return h

        active_actions = action_sequence.index_select(0, active_idx)[:, :active_max_steps]
        active_z = z.index_select(0, active_idx).unsqueeze(1).expand(-1, active_max_steps, -1)
        gru_input = torch.cat([active_actions, active_z], dim=-1)

        sorted_lengths, sort_order = torch.sort(active_lengths, descending=True)
        gru_input = gru_input.index_select(0, sort_order)
        h0 = h.index_select(0, active_idx).index_select(0, sort_order).unsqueeze(0)

        packed = pack_padded_sequence(
            gru_input,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, h_sorted = self.gru(packed, h0)

        _, unsort_order = torch.sort(sort_order)
        h_active = h_sorted.squeeze(0).index_select(0, unsort_order)

        updated_h = h.clone()
        updated_h[active_idx] = h_active
        return updated_h
