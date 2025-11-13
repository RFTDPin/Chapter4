"""Estimate AIS↔video time offset via cross-correlation."""
from __future__ import annotations

import argparse
from typing import List

import numpy as np

from . import utils


def compute_offset(
    ais: List[utils.AISTrack], frames: List[utils.FrameRecord], search_range: float
) -> float:
    frame_ts = np.array([f.timestamp for f in frames], dtype=float)
    ais_ts = np.array([a.timestamp for a in ais], dtype=float)
    ais_speed = np.array([a.sog for a in ais], dtype=float)
    frame_speed = np.interp(frame_ts, ais_ts, ais_speed, left=ais_speed[0], right=ais_speed[-1])
    offsets = np.linspace(-search_range, search_range, num=401)
    best_offset = 0.0
    best_score = -np.inf
    for offset in offsets:
        shifted = np.interp(frame_ts + offset, ais_ts, ais_speed, left=np.nan, right=np.nan)
        valid = ~np.isnan(shifted)
        if valid.sum() < 5:
            continue
        score = np.corrcoef(frame_speed[valid], shifted[valid])[0, 1]
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate AIS-video time offset")
    parser.add_argument("--ais", required=True)
    parser.add_argument("--frame-index", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--search-range", type=float, default=5.0, help="seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ais = utils.load_ais(args.ais)
    frames = utils.load_frame_index(args.frame_index)
    offset = compute_offset(ais, frames, search_range=args.search_range)
    payload = {
        "scene_id": args.scene_id,
        "fps": args.fps,
        "search_range": args.search_range,
        "time_offset": offset,
        "ais_samples": len(ais),
        "frame_samples": len(frames),
    }
    utils.save_json(args.output, payload)


if __name__ == "__main__":
    main()
