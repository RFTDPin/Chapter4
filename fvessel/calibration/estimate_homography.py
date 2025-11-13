"""Estimate water-plane homography for FVessel scenes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from . import utils


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    scale = np.sqrt(2) / (std + 1e-8)
    t = np.array(
        [
            [scale[0], 0, -scale[0] * mean[0]],
            [0, scale[1], -scale[1] * mean[1]],
            [0, 0, 1],
        ]
    )
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1).T
    norm_points = (t @ points_h).T
    return norm_points[:, :2], t


def dlt(pixels: np.ndarray, world: np.ndarray) -> np.ndarray:
    pix_n, t_pix = normalize_points(pixels)
    world_n, t_world = normalize_points(world)
    n = len(pixels)
    a = []
    for i in range(n):
        x, y = world_n[i]
        u, v = pix_n[i]
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    a = np.asarray(a)
    _, _, vh = np.linalg.svd(a)
    h_norm = vh[-1].reshape(3, 3)
    h = np.linalg.inv(t_pix) @ h_norm @ t_world
    return h / h[2, 2]


def estimate_h(payload_path: Path, output_path: Path, scene_id: str, img_shape: tuple[int, int]) -> Dict:
    pixels, world = utils.load_keypoint_matches(payload_path)
    h = dlt(pixels, world)
    summary = {
        "scene_id": scene_id,
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "homography": h.tolist(),
        "num_points": len(pixels),
        "source_file": str(payload_path),
    }
    utils.save_json(output_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate homography for FVessel scene")
    parser.add_argument("--matches", required=True, help="JSON/CSV file with pixel-world matches")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True, help="Path to save homography json")
    parser.add_argument("--img-width", type=int, required=True)
    parser.add_argument("--img-height", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimate_h(
        payload_path=Path(args.matches),
        output_path=Path(args.output),
        scene_id=args.scene_id,
        img_shape=(args.img_width, args.img_height),
    )


if __name__ == "__main__":
    main()
