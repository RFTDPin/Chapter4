
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict


def _select_dims(arr: np.ndarray, idxs: Tuple[int, ...]) -> np.ndarray:
    """选择特征维度；支持 [T, D] 或 [N, T, D] 两种形状。"""
    if arr.ndim == 3:
        return arr[..., list(idxs)]
    elif arr.ndim == 2:
        return arr[:, list(idxs)]
    else:
        raise ValueError(f"Unsupported ndim {arr.ndim} for feature selection")


def _variant_indices(kind: str, variant: str) -> Tuple[int, ...]:
    """
    返回不同输入形态的维度选择：
      kind='vis'：视频侧特征顺序 [x, y, w, h, c, s, psi, a, dpsi, r]
      kind='ais'：AIS侧特征顺序 [u, v, s, theta, ds, dtheta, sin_dt, cos_dt]
    variant:
      - 'kine'：含显式动力学（全维）
      - 'raw' ：仅基础几何与速度/航向（对照组A）
    """
    if kind == 'vis':
        if variant == 'kine':
            return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        elif variant == 'raw':
            return (0, 1, 2, 3, 4)
        else:
            raise ValueError(f"Unknown variant {variant}")
    elif kind == 'ais':
        if variant == 'kine':
            return (0, 1, 2, 3, 4, 5, 6, 7)
        elif variant == 'raw':
            return (0, 1, 2, 3)
        else:
            raise ValueError(f"Unknown variant {variant}")
    else:
        raise ValueError(f"Unknown kind {kind}")


class ContrastivePairsDataset(Dataset):
    """
    从 .npz 中读取配对样本（正样本对：同一船舶的 video/AIS 轨迹）。
    __getitem__ 返回一个字典：
n      {
        'vis_seq':  FloatTensor [T_vis, D_vis],
        'vis_mask': BoolTensor  [T_vis],
        'ais_seq':  FloatTensor [T_ais, D_ais],
        'ais_mask': BoolTensor  [T_ais],
        'vis_pos':  FloatTensor [T_vis, 2] (optional),
        'ais_pos':  FloatTensor [T_ais, 2] (optional),
        'id':       int
      }
    """
    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        feature_variant: str = "kine",
        return_pos: bool = False,
        mmap_mode: Optional[str] = None
    ):
        super().__init__()
        self.path = npz_path
        self.split = split
        self.variant = feature_variant
        self.return_pos = return_pos
        self.mmap_mode = mmap_mode

        self._load_arrays()

    def _load_arrays(self) -> None:
        """加载 .npz 中的数组。"""
        self.store = np.load(self.path, allow_pickle=False, mmap_mode=self.mmap_mode)
        prefix = self.split + "_"
        # # 直接内存映射，避免一次性加载全部数据
        # self.store = np.load(self.path, allow_pickle=False, mmap_mode=mmap_mode)
        # prefix = split + "_"

        # 必需字段
        self.vis_seq = self.store[prefix + "vis_seq"]   # [N, T_vis, D_vis_full]
        self.vis_mask = self.store[prefix + "vis_mask"] # [N, T_vis]
        self.ais_seq = self.store[prefix + "ais_seq"]   # [N, T_ais, D_ais_full]
        self.ais_mask = self.store[prefix + "ais_mask"] # [N, T_ais]
        self.ids = self.store[prefix + "ids"]           # [N]

        # 评估可选字段（用于 MOFP 等定位误差）
        self.vis_pos = self.store.get(prefix + "vis_pos", None)
        self.ais_pos = self.store.get(prefix + "ais_pos", None)

        # 记录要选择的特征维
        self.vis_idx = _variant_indices('vis', self.variant)
        self.ais_idx = _variant_indices('ais', self.variant)

        # 记录时长
        self.T_vis = self.vis_seq.shape[1]
        self.T_ais = self.ais_seq.shape[1]

    def __getstate__(self):
        state = self.__dict__.copy()
        # np.load 返回的 NpzFile 内部持有无法被 pickle 的文件句柄，
        # 需要在 worker 进程中重新打开
        state['store'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_arrays()

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 选取特征维并转换为 torch.Tensor
        vis = _select_dims(self.vis_seq[idx], self.vis_idx).astype(np.float32)  # [T_vis, Dv]
        ais = _select_dims(self.ais_seq[idx], self.ais_idx).astype(np.float32)  # [T_ais, Da]
        vmask = self.vis_mask[idx].astype(bool)
        amask = self.ais_mask[idx].astype(bool)

        item = {
            "vis_seq":  torch.from_numpy(vis),     # [T_vis, Dv]
            "vis_mask": torch.from_numpy(vmask),   # [T_vis]
            "ais_seq":  torch.from_numpy(ais),     # [T_ais, Da]
            "ais_mask": torch.from_numpy(amask),   # [T_ais]
            "id":       int(self.ids[idx]),
        }

        if self.return_pos and (self.vis_pos is not None) and (self.ais_pos is not None):
            item["vis_pos"] = torch.from_numpy(self.vis_pos[idx].astype(np.float32))
            item["ais_pos"] = torch.from_numpy(self.ais_pos[idx].astype(np.float32))

        return item


def build_loader(
    npz_path: str,
    split: str,
    batch_size: int = 128,
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    feature_variant: str = "kine",
    return_pos: bool = False,
    pin_memory: bool = True,
    drop_last: bool = True,
    mmap_mode: Optional[str] = None
) -> DataLoader:
    if shuffle is None:
        shuffle = (split == "train")
    ds = ContrastivePairsDataset(
        npz_path=npz_path, split=split, feature_variant=feature_variant,
        return_pos=return_pos, mmap_mode=mmap_mode
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )
