
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerSeqEncoder(nn.Module):
    """
    Lightweight Transformer encoder with Pre-LN and CLS pooling.
    """
    def __init__(self, in_dim, d_model=64, n_layers=3, n_heads=4, d_ff=128,
                 pooling='cls', max_len=1024):
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = PositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x: [B,T,D], mask: [B,T] (True for valid)
        B, T, _ = x.size()
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+T, D]
        x = self.pos(x)

        # key_padding_mask: True for padding (invalid)
        if mask is not None:
            kpm = torch.cat([torch.ones(B, 1, device=x.device, dtype=torch.bool), ~mask], dim=1)
        else:
            kpm = None

        x = self.encoder(x, src_key_padding_mask=kpm)
        x = self.norm(x)

        if self.pooling == 'cls':
            z = x[:, 0, :]  # [B,D]
        else:
            # mean over valid steps
            if mask is not None:
                x_no_cls = x[:, 1:, :]
                denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
                z = (x_no_cls * mask.unsqueeze(-1).float()).sum(dim=1) / denom
            else:
                z = x[:, 1:, :].mean(dim=1)
        return z
