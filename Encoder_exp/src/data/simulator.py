
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any


@dataclass
class SimConfig:
    n_train: int = 12000
    n_val: int = 1500
    n_test: int = 1500
    seed: int = 2025

    T_vis_max: int = 64
    T_ais_max: int = 24

    dt_vis: float = 1.0
    ais_dt_range: Tuple[float, float] = (2.0, 8.0)
    v_range: Tuple[float, float] = (2.0, 6.0)
    turn_prob: float = 0.2

    pix_noise_vis: float = 1.5
    pos_noise_ais: float = 3.0
    miss_vis_prob: float = 0.06
    miss_ais_prob: float = 0.04
    occ_len_range: Tuple[int, int] = (2, 6)

    dt_enc_omega: float = 0.5


class TrajectorySimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _rng(self, seed=None):
        return np.random.default_rng(self.rng.integers(1, 10**9) if seed is None else seed)

    def simulate_sample(self, seed: int = None) -> Dict[str, Any]:
        cfg = self.cfg
        rng = self._rng(seed)

        T = cfg.T_vis_max
        t_vis = np.arange(0, T * cfg.dt_vis, cfg.dt_vis)

        p0 = rng.uniform([0, 0], [640, 360])
        theta0 = rng.uniform(-np.pi, np.pi)
        v = rng.uniform(*cfg.v_range)

        turns = rng.random(T) < cfg.turn_prob
        dtheta = rng.normal(0, 0.05, T) * turns
        theta_seq = np.cumsum(dtheta) + theta0

        disp = np.stack([v * np.cos(theta_seq) * cfg.dt_vis,
                         v * np.sin(theta_seq) * cfg.dt_vis], axis=1)
        P_vis = np.cumsum(np.vstack([p0, disp[:-1]]), axis=0)

        P_vis_noisy = P_vis + rng.normal(0, cfg.pix_noise_vis, P_vis.shape)
        vis_mask = rng.random(T) > cfg.miss_vis_prob
        if rng.random() < 0.5:
            L = rng.integers(cfg.occ_len_range[0], cfg.occ_len_range[1] + 1)
            s0 = rng.integers(3, max(4, T - 3 - L))  # safe bounds
            vis_mask[s0:s0 + L] = False

        ais_ts = [0.0]
        while ais_ts[-1] < T * cfg.dt_vis:
            dt = rng.uniform(*cfg.ais_dt_range)
            ais_ts.append(ais_ts[-1] + dt)
        ais_ts = np.array(ais_ts[:-1])
        L_ais = len(ais_ts)

        ais_idx = np.clip((ais_ts / cfg.dt_vis).astype(int), 0, T - 1)
        P_ais = P_vis[ais_idx] + rng.normal(0, cfg.pos_noise_ais, (L_ais, 2))

        dt_arr = np.diff(np.hstack([[0.0], ais_ts]))
        dt_arr[dt_arr == 0] = 1e-3
        dP = np.diff(P_ais, axis=0, prepend=P_ais[:1])
        s_ais = np.linalg.norm(dP, axis=1) / dt_arr
        theta_ais = np.arctan2(dP[:, 1], dP[:, 0])

        ais_mask = rng.random(L_ais) > cfg.miss_ais_prob
        P_ais[~ais_mask] = np.nan
        s_ais[~ais_mask] = np.nan
        theta_ais[~ais_mask] = np.nan

        return dict(
            vis=dict(t=t_vis, pos=P_vis_noisy, mask=vis_mask.astype(bool)),
            ais=dict(t=ais_ts, pos=P_ais, speed=s_ais, heading=theta_ais, mask=ais_mask.astype(bool)),
            truth=dict(P_vis=P_vis)
        )

    @staticmethod
    def _diff_with_nans(x: np.ndarray) -> np.ndarray:
        x0 = np.where(np.isfinite(x), x, np.nan)
        for i in range(1, len(x0)):
            if not np.isfinite(x0[i]).all():
                x0[i] = x0[i - 1]
        dx = np.diff(x0, axis=0, prepend=x0[:1])
        return dx

    def build_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        # Video
        t_v = sample['vis']['t']
        P_v = sample['vis']['pos'].astype(np.float32)
        mask_v = sample['vis']['mask'].astype(bool)

        Tv = len(t_v)
        w = np.full(Tv, 40.0, dtype=np.float32)
        h = np.full(Tv, 20.0, dtype=np.float32)
        c = np.where(mask_v, 0.9, 0.0).astype(np.float32)

        dP = self._diff_with_nans(P_v)
        dtv = np.diff(np.hstack([[0.0], t_v]))
        dtv[dtv == 0] = 1e-3
        s = (np.linalg.norm(dP, axis=1) / dtv).astype(np.float32)
        psi = np.arctan2(dP[:, 1], dP[:, 0]).astype(np.float32)
        a = np.diff(s, prepend=s[:1]).astype(np.float32)
        dpsi = np.diff(psi, prepend=psi[:1]).astype(np.float32)
        r = np.diff(np.log(w * h), prepend=np.log(w[0] * h[0])).astype(np.float32)

        vis_feat = np.stack([
            P_v[:, 0], P_v[:, 1], w, h, c, s, psi, a, dpsi, r
        ], axis=1)
        vis_feat = np.nan_to_num(vis_feat, nan=0.0)
        vis_pos = P_v.copy()

        vis_feat, vis_mask, vis_pos = self._pad_clip_triplet(
            vis_feat, mask_v, vis_pos, cfg.T_vis_max
        )

        # AIS
        t_a = sample['ais']['t']
        P_a = sample['ais']['pos'].astype(np.float32)
        s_a = sample['ais']['speed'].astype(np.float32)
        th_a = sample['ais']['heading'].astype(np.float32)
        mask_a = sample['ais']['mask'].astype(bool)

        dsa = np.diff(s_a, prepend=s_a[:1]).astype(np.float32)
        dha = np.diff(th_a, prepend=th_a[:1]).astype(np.float32)
        dt_a = np.diff(np.hstack([[0.0], t_a]))
        dt_a[dt_a == 0] = 1e-3
        gamma = np.stack([
            np.sin(cfg.dt_enc_omega * dt_a),
            np.cos(cfg.dt_enc_omega * dt_a)
        ], axis=1).astype(np.float32)

        ais_feat = np.stack([
            P_a[:, 0], P_a[:, 1], s_a, th_a, dsa, dha, gamma[:, 0], gamma[:, 1]
        ], axis=1)
        ais_feat = np.nan_to_num(ais_feat, nan=0.0)
        ais_pos = P_a.copy()

        ais_feat, ais_mask, ais_pos = self._pad_clip_triplet(
            ais_feat, mask_a, ais_pos, cfg.T_ais_max
        )

        return dict(
            vis_seq=vis_feat.astype(np.float32),
            vis_mask=vis_mask.astype(bool),
            vis_pos=vis_pos.astype(np.float32),
            ais_seq=ais_feat.astype(np.float32),
            ais_mask=ais_mask.astype(bool),
            ais_pos=ais_pos.astype(np.float32)
        )

    @staticmethod
    def _pad_clip_triplet(feat: np.ndarray, mask: np.ndarray, pos: np.ndarray, T: int):
        L = feat.shape[0]
        if L >= T:
            feat_out = feat[:T].copy()
            mask_out = mask[:T].copy()
            pos_out = pos[:T].copy()
        else:
            pad_len = T - L
            feat_out = np.vstack([feat, np.zeros((pad_len, feat.shape[1]), dtype=feat.dtype)])
            mask_out = np.hstack([mask, np.zeros((pad_len,), dtype=bool)])
            pos_pad = np.full((pad_len, pos.shape[1]), np.nan, dtype=pos.dtype)
            pos_out = np.vstack([pos, pos_pad])
        return feat_out, mask_out, pos_out

    def simulate_split(self, n_samples: int, seed: int) -> Dict[str, np.ndarray]:
        rng_b = self._rng(seed)
        vis_seq_list, vis_mask_list, vis_pos_list = [], [], []
        ais_seq_list, ais_mask_list, ais_pos_list = [], [], []
        ids = []

        for i in range(n_samples):
            s = self.simulate_sample(seed=int(rng_b.integers(1, 10**9)))
            f = self.build_features(s)
            vis_seq_list.append(f["vis_seq"])
            vis_mask_list.append(f["vis_mask"])
            vis_pos_list.append(f["vis_pos"])
            ais_seq_list.append(f["ais_seq"])
            ais_mask_list.append(f["ais_mask"])
            ais_pos_list.append(f["ais_pos"])
            ids.append(i)

        vis_seq = np.stack(vis_seq_list, axis=0)
        vis_mask = np.stack(vis_mask_list, axis=0)
        vis_pos = np.stack(vis_pos_list, axis=0)
        ais_seq = np.stack(ais_seq_list, axis=0)
        ais_mask = np.stack(ais_mask_list, axis=0)
        ais_pos = np.stack(ais_pos_list, axis=0)
        ids = np.array(ids, dtype=np.int64)

        return dict(
            vis_seq=vis_seq, vis_mask=vis_mask, vis_pos=vis_pos,
            ais_seq=ais_seq, ais_mask=ais_mask, ais_pos=ais_pos,
            ids=ids
        )

    def save_npz(self, splits: Dict[str, Dict[str, np.ndarray]], out_path: str):
        arrays = {}
        for split, pack in splits.items():
            for k, v in pack.items():
                arrays[f"{split}_{k}"] = v
        np.savez_compressed(out_path, **arrays)
