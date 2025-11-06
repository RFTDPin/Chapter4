
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Symmetric NT-Xent (InfoNCE) loss for cross-modal pairs.
    Given z_v, z_a in R^{Bxd} (assumed L2-normalized), uses in-batch negatives.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        # z_v, z_a: [B, d]
        logits = torch.matmul(z_v, z_a.t()) / self.tau  # [B,B]
        labels = torch.arange(z_v.size(0), device=z_v.device)
        loss_v2a = torch.nn.functional.cross_entropy(logits, labels)
        loss_a2v = torch.nn.functional.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_v2a + loss_a2v)
