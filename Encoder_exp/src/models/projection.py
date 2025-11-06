
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    LN -> MLP -> L2 normalize
    """
    def __init__(self, in_dim, proj_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim)
        )

    def forward(self, z):
        z = self.net(z)
        z = F.normalize(z, dim=-1)
        return z
