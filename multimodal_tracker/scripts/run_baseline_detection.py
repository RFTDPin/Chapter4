"""Run YOLOv11 baseline on FVessel data and export metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install ultralytics to run baseline detection") from exc


def run_eval(weights: str, data_cfg: str, split: str, imgsz: int, batch: int) -> Dict:
    model = YOLO(weights)
    results = model.val(data=data_cfg, split=split, imgsz=imgsz, batch=batch, conf=0.25, iou=0.6)
    metrics = {
        "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
        "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "map5095": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "maps": float(results.results_dict.get("metrics/mAP50(S)", 0.0)),
        "speed_infer_ms": float(results.speed.get("inference", 0.0)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO baseline on FVessel")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="YOLO data yaml path")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", default="results/tables/baseline_detection.csv")
    args = parser.parse_args()

    metrics = run_eval(args.weights, args.data, args.split, args.imgsz, args.batch)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"Saved baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
