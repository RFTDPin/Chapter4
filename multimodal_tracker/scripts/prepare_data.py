"""Prepare FVessel priors by aligning AIS with video frames."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from fvessel.calibration import estimate_homography, estimate_time_offset, utils


def build_priors(
    frames: List[utils.FrameRecord],
    ais_tracks: Dict[int, List[utils.AISTrack]],
    homography: np.ndarray,
    time_offset: float,
    lon0: float,
    lat0: float,
    sigma_base: float,
    sigma_speed: float,
    confidence: float,
) -> Iterable[Dict]:
    for mmsi, track in ais_tracks.items():
        sorted_track = sorted(track, key=lambda t: t.timestamp)
        for frame in frames:
            query_ts = frame.timestamp + time_offset
            sample = utils.interpolate_track(sorted_track, query_ts)
            if sample is None:
                continue
            sigma = sigma_base + sigma_speed * abs(sample.sog)
            x, y = utils.geodetic_to_local_xy(sample.lon, sample.lat, lon0=lon0, lat0=lat0)
            uv = utils.apply_homography(homography, np.array([[x, y]]))[0]
            yield {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "mmsi": mmsi,
                "u": float(uv[0]),
                "v": float(uv[1]),
                "sigma_r": float(sigma),
                "speed": float(sample.sog),
                "course": float(sample.cog),
                "confidence": float(confidence),
                "source": "ais_alignment",
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FVessel priors and calibration")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--dataset-root", default="fvessel/raw")
    parser.add_argument("--output-root", default="fvessel")
    parser.add_argument("--matches", help="Pixel/world correspondences for homography", required=False)
    parser.add_argument("--homography-json", help="Existing homography json")
    parser.add_argument("--time-offset-json", help="Existing time offset json")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--img-width", type=int, required=True)
    parser.add_argument("--img-height", type=int, required=True)
    parser.add_argument("--sigma-base", type=float, default=6.0)
    parser.add_argument("--sigma-speed", type=float, default=0.2)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--lon0", type=float, required=True)
    parser.add_argument("--lat0", type=float, required=True)
    parser.add_argument("--search-range", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    scene_root = dataset_root / args.scene_id
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ais_path = scene_root / "ais.csv"
    frame_index_path = scene_root / "frame_index.csv"
    matches_path = Path(args.matches) if args.matches else scene_root / "homography_pairs.json"

    homography_json = Path(args.homography_json) if args.homography_json else output_root / f"calib/{args.scene_id}_homography.json"
    time_offset_json = Path(args.time_offset_json) if args.time_offset_json else output_root / f"calib/{args.scene_id}_time_offset.json"

    homography_json.parent.mkdir(parents=True, exist_ok=True)

    if not homography_json.exists():
        if not matches_path.exists():
            raise FileNotFoundError(f"Missing matches file: {matches_path}")
        estimate_homography.estimate_h(
            payload_path=matches_path,
            output_path=homography_json,
            scene_id=args.scene_id,
            img_shape=(args.img_width, args.img_height),
        )

    ais = utils.load_ais(ais_path)
    frames = utils.load_frame_index(frame_index_path)

    if not time_offset_json.exists():
        offset = estimate_time_offset.compute_offset(
            ais=ais,
            frames=frames,
            search_range=args.search_range,
        )
        utils.save_json(
            time_offset_json,
            {
                "scene_id": args.scene_id,
                "fps": args.fps,
                "search_range": args.search_range,
                "time_offset": offset,
                "ais_samples": len(ais),
                "frame_samples": len(frames),
            },
        )
    else:
        offset = json.loads(time_offset_json.read_text())["time_offset"]

    homography_payload = json.loads(homography_json.read_text())
    homography = np.asarray(homography_payload["homography"], dtype=float)

    tracks = defaultdict(list)
    for record in ais:
        tracks[record.mmsi].append(record)

    priors_path = output_root / f"priors/{args.scene_id}.jsonl"
    priors_path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_jsonl(
        priors_path,
        build_priors(
            frames=frames,
            ais_tracks=tracks,
            homography=homography,
            time_offset=offset,
            lon0=args.lon0,
            lat0=args.lat0,
            sigma_base=args.sigma_base,
            sigma_speed=args.sigma_speed,
            confidence=args.confidence,
        ),
    )

    print(f"Saved priors to {priors_path}")
