
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn_encoder import TCNBlock, Gating


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class HybridSeqEncoder(nn.Module):
    """
    Conv (local dynamics) + Transformer (global context).
    - Optional kinematic gating on selected input channels.
    - TCN frontend: residual dilated conv blocks for local smoothing/derivatives.
    - Transformer backend: global aggregation (mean/cls pooling).

    Forward:
        x: [B, T, D], mask: [B, T] (True for valid)
        -> z: [B, d_model]
    """
    def __init__(self,
                 in_dim: int,
                 # Frontend (TCN)
                 front_hidden: int = 64,
                 front_layers: int = 2,
                 front_dilations=(1, 2),
                 use_gating: bool = True,
                 gate_idx=None,
                 # Backend (Transformer)
                 d_model: int = 64,
                 tf_layers: int = 2,
                 n_heads: int = 4,
                 d_ff: int = 128,
                 dropout: float = 0.1,
                 pooling: str = "mean",
                 max_len: int = 1024):
        super().__init__()
        self.in_dim = in_dim
        self.pooling = pooling

        # Gating on raw inputs (explicit kinematics)
        self.gate = Gating(in_dim, gate_idx=gate_idx) if use_gating and gate_idx else None
        if self.gate is not None:
            # 更积极地保留动力学量：bias 初始化为一个正数，避免早期抑制
            nn.init.constant_(self.gate.gate.bias, 2.0)

        # TCN frontend
        self.front_proj = nn.Linear(in_dim, front_hidden)
        self.front_blocks = nn.ModuleList([
            TCNBlock(front_hidden, dilation=front_dilations[i % len(front_dilations)], kernel_size=3)
            for i in range(front_layers)
        ])
        self.front_norm = nn.LayerNorm(front_hidden)

        # Bridge to Transformer
        self.bridge = nn.Linear(front_hidden, d_model) if front_hidden != d_model else nn.Identity()

        # Transformer backend
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.back_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x: [B,T,D], mask: [B,T] True for valid
        if self.gate is not None:
            x = self.gate(x)
        x = self.front_proj(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        for blk in self.front_blocks:
            x = blk(x, mask=mask)  # [B,T,H]

        x = self.front_norm(x)
        x = self.bridge(x)  # [B,T,d_model]

        B, T, D = x.shape
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)  # [B,1+T,D]

        x = self.pos(x)

        # key_padding_mask: True for padding (invalid). CLS 不屏蔽。
        if mask is not None:
            if self.pooling == "cls":
                kpm = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), ~mask], dim=1)
            else:
                kpm = ~mask
        else:
            kpm = None

        x = self.encoder(x, src_key_padding_mask=kpm)
        x = self.back_norm(x)

        if self.pooling == "cls":
            z = x[:, 0, :]
        else:
            if mask is not None:
                denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
                z = (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom
            else:
                z = x.mean(dim=1)
        return z
