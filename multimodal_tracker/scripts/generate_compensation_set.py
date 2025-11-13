"""Generate compensation set indices from AIS priors and YOLO detections."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_priors(path: Path) -> Dict[int, List[Dict]]:
    entries: Dict[int, List[Dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            entries[payload["frame_id"]].append(payload)
    return entries


def load_detections(path: Path) -> Dict[int, List[Dict]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    # expected structure: {"frame_id": [{"bbox": [x1,y1,x2,y2], "score": 0.9, "label": "ship"}, ...]}
    return {int(k): v for k, v in data.items()}


def iou(b1: List[float], b2: List[float]) -> float:
    xa = max(b1[0], b2[0])
    ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct compensation set")
    parser.add_argument("--priors", required=True)
    parser.add_argument("--detections", required=True, help="JSON per frame detections")
    parser.add_argument("--radius", type=float, default=25.0)
    parser.add_argument("--iou-th", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    priors = load_priors(Path(args.priors))
    detections = load_detections(Path(args.detections))

    misses = []
    for frame_id, prior_list in priors.items():
        dets = detections.get(frame_id, [])
        for prior in prior_list:
            px = prior["u"]
            py = prior["v"]
            matched = False
            for det in dets:
                bbox = det.get("bbox") or det.get("xyxy")
                if bbox is None:
                    continue
                cx = 0.5 * (bbox[0] + bbox[2])
                cy = 0.5 * (bbox[1] + bbox[3])
                if np.hypot(cx - px, cy - py) <= args.radius:
                    matched = True
                    break
            if not matched:
                misses.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": prior["timestamp"],
                        "mmsi": prior["mmsi"],
                        "u": prior["u"],
                        "v": prior["v"],
                        "sigma_r": prior["sigma_r"],
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"misses": misses}, fp, indent=2)
    print(f"Saved {len(misses)} missed cases to {output_path}")


if __name__ == "__main__":
    main()
