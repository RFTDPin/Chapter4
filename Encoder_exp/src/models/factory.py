
from typing import Dict
import torch.nn as nn

from .tcn_encoder import KinematicTCNEncoder
from .transformer_encoder import TransformerSeqEncoder
from .projection import ProjectionHead


class DualEncoder(nn.Module):
    def __init__(self, video_enc: nn.Module, ais_enc: nn.Module, proj_dim: int = 64, z_dim: int = 64):
        super().__init__()
        self.video_encoder = video_enc
        self.ais_encoder = ais_enc
        self.proj_v = ProjectionHead(in_dim=z_dim, proj_dim=proj_dim)
        self.proj_a = ProjectionHead(in_dim=z_dim, proj_dim=proj_dim)

    def forward(self, vis_seq, vis_mask, ais_seq, ais_mask):
        z_v = self.video_encoder(vis_seq, vis_mask)  # [B, z_dim]
        z_a = self.ais_encoder(ais_seq, ais_mask)    # [B, z_dim]
        z_v = self.proj_v(z_v)
        z_a = self.proj_a(z_a)
        return z_v, z_a


def build_dual_encoder(cfg: Dict) -> nn.Module:
    """
    cfg:
      type: 'exp' | 'trans_raw' | 'trans_kine'
      proj_dim: int
      video: {in_dim, hidden/layers/dilations or d_model/n_layers/n_heads/d_ff}
      ais:   {in_dim, ...}
    """
    model_type = cfg["type"]
    proj_dim = cfg.get("proj_dim", 64)

    if model_type == "exp":
        # Indices for kinematic gating (video: s, psi, a, dpsi, r -> indices 5..9; ais: speed/heading etc. -> 2..5)
        v_gate_idx = list(range(5, 10))  # on video features [s,psi,a,dpsi,r]
        a_gate_idx = list(range(2, 6))   # on ais features [s,theta,ds,dtheta]

        vconf = cfg["video"]; aconf = cfg["ais"]
        v_enc = KinematicTCNEncoder(
            in_dim=vconf["in_dim"],
            hidden=vconf.get("hidden", 64),
            layers=vconf.get("layers", 3),
            dilations=tuple(vconf.get("dilations", [1,2,4])),
            use_gating=vconf.get("use_gating", True),
            attn_pool=vconf.get("attn_pool", True),
            gate_idx=v_gate_idx if vconf["in_dim"] >= 10 else []
        )
        a_enc = KinematicTCNEncoder(
            in_dim=aconf["in_dim"],
            hidden=aconf.get("hidden", 64),
            layers=aconf.get("layers", 3),
            dilations=tuple(aconf.get("dilations", [1,2,4])),
            use_gating=aconf.get("use_gating", True),
            attn_pool=aconf.get("attn_pool", True),
            gate_idx=a_gate_idx if aconf["in_dim"] >= 6 else []
        )
        z_dim = vconf.get("hidden", 64)

    elif model_type in ["trans_raw", "trans_kine"]:
        vconf = cfg["video"]; aconf = cfg["ais"]
        v_enc = TransformerSeqEncoder(
            in_dim=vconf["in_dim"],
            d_model=vconf.get("d_model", 64),
            n_layers=vconf.get("n_layers", 3),
            n_heads=vconf.get("n_heads", 4),
            d_ff=vconf.get("d_ff", 128),
            pooling=vconf.get("pooling", "cls"),
            max_len=max(1024, vconf.get("max_len", 1024))
        )
        a_enc = TransformerSeqEncoder(
            in_dim=aconf["in_dim"],
            d_model=aconf.get("d_model", 64),
            n_layers=aconf.get("n_layers", 3),
            n_heads=aconf.get("n_heads", 4),
            d_ff=aconf.get("d_ff", 128),
            pooling=aconf.get("pooling", "cls"),
            max_len=max(1024, aconf.get("max_len", 1024))
        )
        z_dim = vconf.get("d_model", 64)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = DualEncoder(v_enc, a_enc, proj_dim=proj_dim, z_dim=z_dim)
    return model
