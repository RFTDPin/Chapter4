
import torch.nn as nn
from typing import Dict

from .tcn_encoder import KinematicTCNEncoder
from .transformer_encoder import TransformerSeqEncoder
from .projection import ProjectionHead
from .hybrid_encoder import HybridSeqEncoder


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
    model_type = cfg["type"]
    proj_dim = cfg.get("proj_dim", 64)

    if model_type == "exp":
        vconf = cfg["video"]; aconf = cfg["ais"]
        v_enc = KinematicTCNEncoder(
            in_dim=vconf["in_dim"],
            hidden=vconf.get("hidden", 64),
            layers=vconf.get("layers", 3),
            dilations=tuple(vconf.get("dilations", [1,2,4])),
            use_gating=vconf.get("use_gating", True),
            attn_pool=vconf.get("attn_pool", True),
            gate_idx=list(range(5,10)) if vconf["in_dim"] >= 10 else []
        )
        a_enc = KinematicTCNEncoder(
            in_dim=aconf["in_dim"],
            hidden=aconf.get("hidden", 64),
            layers=aconf.get("layers", 3),
            dilations=tuple(aconf.get("dilations", [1,2,4])),
            use_gating=aconf.get("use_gating", True),
            attn_pool=aconf.get("attn_pool", True),
            gate_idx=list(range(2,6)) if aconf["in_dim"] >= 6 else []
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

    elif model_type in ["hybrid_kine", "hybrid_raw"]:
        vconf = cfg["video"]; aconf = cfg["ais"]
        v_gate_idx = list(range(5, 10)) if vconf["in_dim"] >= 10 else []
        a_gate_idx = list(range(2, 6))  if aconf["in_dim"] >= 6  else []
        v_enc = HybridSeqEncoder(
            in_dim=vconf["in_dim"],
            front_hidden=vconf.get("front_hidden", 64),
            front_layers=vconf.get("front_layers", 2),
            front_dilations=tuple(vconf.get("front_dilations", [1,2])),
            use_gating=vconf.get("use_gating", True),
            gate_idx=v_gate_idx,
            d_model=vconf.get("d_model", 64),
            tf_layers=vconf.get("tf_layers", 2),
            n_heads=vconf.get("n_heads", 4),
            d_ff=vconf.get("d_ff", 128),
            dropout=vconf.get("dropout", 0.1),
            pooling=vconf.get("pooling", "mean"),
        )
        a_enc = HybridSeqEncoder(
            in_dim=aconf["in_dim"],
            front_hidden=aconf.get("front_hidden", 64),
            front_layers=aconf.get("front_layers", 2),
            front_dilations=tuple(aconf.get("front_dilations", [1,2])),
            use_gating=aconf.get("use_gating", True),
            gate_idx=a_gate_idx,
            d_model=aconf.get("d_model", 64),
            tf_layers=aconf.get("tf_layers", 2),
            n_heads=aconf.get("n_heads", 4),
            d_ff=aconf.get("d_ff", 128),
            dropout=aconf.get("dropout", 0.1),
            pooling=aconf.get("pooling", "mean"),
        )
        z_dim = vconf.get("d_model", 64)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return DualEncoder(v_enc, a_enc, proj_dim=proj_dim, z_dim=z_dim)
