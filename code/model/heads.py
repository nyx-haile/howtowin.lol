import torch
import torch.nn as nn


class NextEventHead(nn.Module):
    """Multi-hot logits over event types for the next-minute window."""
    def __init__(self, d_z: int, n_event_types: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_event_types),
        )

    def forward(self, z):
        return self.net(z)


class OutcomeHead(nn.Module):
    """Scalar logit for game-win at each anchor."""
    def __init__(self, d_z: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class NextDecisionHead(nn.Module):
    """Per-participant categorical over decision types."""
    def __init__(self, d_z: int, n_decisions: int, n_participants: int = 10,
                 d_hidden: int = 256):
        super().__init__()
        self.n_participants = n_participants
        self.n_decisions = n_decisions
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_participants * n_decisions),
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(z.size(0), self.n_participants, self.n_decisions)


class NextFrameHead(nn.Module):
    """Per-participant mu/logvar over frame feature deltas."""
    def __init__(self, d_z: int, n_participants: int = 10, feat_dim: int = 6,
                 d_hidden: int = 256):
        super().__init__()
        self.n_participants = n_participants
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * n_participants * feat_dim),
        )

    def forward(self, z):
        out = self.net(z).view(z.size(0), self.n_participants, self.feat_dim, 2)
        mu = out[..., 0]
        logvar = out[..., 1].clamp(min=-10.0, max=10.0)
        return mu, logvar
