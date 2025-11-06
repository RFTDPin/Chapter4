
import os
import torch
import numpy as np
from typing import Dict

def save_checkpoint(state: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None) -> Dict:
    return torch.load(path, map_location=map_location)

def cosine_similarity_matrix(z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
    """
    z_v, z_a: [N, d], assumed normalized. return [N,N].
    """
    return torch.matmul(z_v, z_a.t())

def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
