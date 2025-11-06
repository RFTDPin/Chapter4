#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick visualization for cmtrack simulated dataset.
Usage:
  python tools/visualize_npz.py --npz data/sim_dataset.npz --split train --idx 0
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_split(npz_path: str, split: str):
    data = np.load(npz_path)
    pack = {
        "vis_seq": data[f"{split}_vis_seq"],   # [N, T_vis, 10]
        "vis_mask": data[f"{split}_vis_mask"], # [N, T_vis]
        "vis_pos": data[f"{split}_vis_pos"],   # [N, T_vis, 2]
        "ais_seq": data[f"{split}_ais_seq"],   # [N, T_ais, 8]
        "ais_mask": data[f"{split}_ais_mask"], # [N, T_ais]
        "ais_pos": data[f"{split}_ais_pos"],   # [N, T_ais, 2]
        "ids": data[f"{split}_ids"],
    }
    return pack

def basic_stats(sample, idx: int):
    vis_mask = sample["vis_mask"][idx].astype(bool)
    ais_mask = sample["ais_mask"][idx].astype(bool)
    stats = {
        "T_vis": int(sample["vis_seq"][idx].shape[0]),
        "T_ais": int(sample["ais_seq"][idx].shape[0]),
        "vis_valid_frames": int(vis_mask.sum()),
        "ais_valid_frames": int(ais_mask.sum()),
        "vis_missing_frames": int((~vis_mask).sum()),
        "ais_missing_frames": int((~ais_mask).sum()),
    }
    return stats

def plot_xy(sample, split: str, idx: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_pos = sample["vis_pos"][idx]            # [T_vis, 2]
    vis_mask = sample["vis_mask"][idx].astype(bool)
    ais_pos = sample["ais_pos"][idx]            # [T_ais, 2]
    ais_mask = sample["ais_mask"][idx].astype(bool)

    v_ok = np.isfinite(vis_pos).all(axis=1) & vis_mask
    a_ok = np.isfinite(ais_pos).all(axis=1) & ais_mask

    fig = plt.figure(figsize=(6,5))
    plt.plot(vis_pos[v_ok,0], vis_pos[v_ok,1], marker='o', linestyle='-', label='Video traj')
    plt.plot(ais_pos[a_ok,0], ais_pos[a_ok,1], marker='x', linestyle='--', label='AIS traj')
    plt.title(f"XY Trajectories ({split}, idx={idx})")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.legend()
    plt.gca().invert_yaxis()   # 图像坐标系更直观
    plt.tight_layout()
    out_path = out_dir / f"{split}_idx{idx}_xy.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_speed(sample, split: str, idx: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_seq = sample["vis_seq"][idx]            # [T_vis, D_vis]; speed列=5
    vis_mask = sample["vis_mask"][idx].astype(bool)
    ais_seq = sample["ais_seq"][idx]            # [T_ais, D_ais]; speed列=2
    ais_mask = sample["ais_mask"][idx].astype(bool)

    vis_speed = vis_seq[:,5]
    ais_speed = ais_seq[:,2]

    fig = plt.figure(figsize=(6,3.8))
    plt.plot(np.arange(len(vis_speed))[vis_mask], vis_speed[vis_mask], marker='o', linestyle='-', label='Video speed')
    plt.plot(np.arange(len(ais_speed))[ais_mask], ais_speed[ais_mask], marker='x', linestyle='--', label='AIS speed')
    plt.title(f"Speed Time Series ({split}, idx={idx})")
    plt.xlabel("time index")
    plt.ylabel("speed (px/s approx.)")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{split}_idx{idx}_speed.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="sim_dataset.npz")
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--idx", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="viz")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}. 请先生成数据: "
                                f"python -m src.data.generate_dataset --config configs/simulate.yaml --out data/sim_dataset.npz")

    pack = load_split(str(npz_path), args.split)
    stats = basic_stats(pack, args.idx)
    print(f"[{args.split}] idx={args.idx} | stats:", stats)

    out_dir = Path(args.outdir)
    p1 = plot_xy(pack, args.split, args.idx, out_dir)
    p2 = plot_speed(pack, args.split, args.idx, out_dir)
    print("Saved figures:")
    print(" -", p1)
    print(" -", p2)

if __name__ == "__main__":
    main()
