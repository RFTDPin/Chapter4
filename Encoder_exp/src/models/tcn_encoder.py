
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gating(nn.Module):
    """
    Feature-wise gating on selected indices (e.g., kinematic components).
    Given x in [B, T, D], produce gate in [B, T, D] with sigmoid and
    apply only to selected dims; others pass through.
    """
    def __init__(self, dim: int, gate_idx=None):
        super().__init__()
        self.dim = dim
        self.gate_idx = gate_idx if gate_idx is not None else list(range(dim))
        self.gate = nn.Linear(len(self.gate_idx), len(self.gate_idx))

        # initialize near-open gates
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        # x: [B, T, D]
        if len(self.gate_idx) == 0:
            return x
        sel = x[..., self.gate_idx]                      # [B,T,|idx|]
        g = torch.sigmoid(self.gate(sel))                # [B,T,|idx|]
        x = x.clone()
        x[..., self.gate_idx] = sel * g
        return x


class TCNBlock(nn.Module):
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, mask=None):
        # x: [B,T,D] -> conv on time => [B,D,T]
        res = x
        xt = x.transpose(1, 2)
        xt = self.conv(xt)
        x2 = xt.transpose(1, 2)
        x2 = self.norm(x2)
        x2 = self.act(x2)
        x = x2 + res
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


class AttentiveTemporalPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        # x: [B,T,D]
        s = self.score(x).squeeze(-1)  # [B,T]
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)    # [B,T]
        z = torch.sum(x * a.unsqueeze(-1), dim=1)  # [B,D]
        return z, a


class KinematicTCNEncoder(nn.Module):
    """
    Residual dilated TCN + kinematic gating + attentive temporal pooling.
    """
    def __init__(self, in_dim, hidden=64, layers=3, dilations=(1,2,4),
                 use_gating=True, attn_pool=True, gate_idx=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers

        self.input_proj = nn.Linear(in_dim, hidden)

        self.use_gating = use_gating
        self.gate = Gating(in_dim, gate_idx=gate_idx) if use_gating else None

        self.blocks = nn.ModuleList([
            TCNBlock(hidden, dilation=dilations[i % len(dilations)], kernel_size=3)
            for i in range(layers)
        ])

        self.attn_pool = AttentiveTemporalPooling(hidden) if attn_pool else None
        self.final_pool = nn.AdaptiveAvgPool1d(1)  # fallback if no attention
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, mask):
        # x: [B,T,D], mask: [B,T] bool
        if self.gate is not None:
            x = self.gate(x)
        x = self.input_proj(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        for blk in self.blocks:
            x = blk(x, mask=mask)

        x = self.norm(x)

        if self.attn_pool is not None:
            z, attn = self.attn_pool(x, mask=mask)
        else:
            # mean over valid steps
            if mask is not None:
                denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
                z = (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom
                attn = None
            else:
                z = x.mean(dim=1)
                attn = None
        return z  # [B,D]
