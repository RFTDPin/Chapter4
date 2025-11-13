"""Utility helpers for FVessel calibration and AIS/video alignment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FrameRecord:
    frame_id: int
    timestamp: float  # seconds since epoch or relative zero


@dataclass
class AISTrack:
    timestamp: float
    mmsi: int
    lon: float
    lat: float
    sog: float  # speed over ground (knots)
    cog: float  # course over ground (degrees)


@dataclass
class PriorRecord:
    frame_id: int
    timestamp: float
    mmsi: int
    u: float
    v: float
    sigma_r: float
    speed: float
    course: float
    confidence: float


EARTH_RADIUS = 6378137.0


def load_keypoint_matches(path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Load pixel↔world correspondences from JSON/CSV file."""
    path = Path(path)
    if path.suffix == ".json":
        entries = json.loads(path.read_text())
        pixels = np.array([[item["u"], item["v"]] for item in entries], dtype=float)
        world = np.array([[item["x"], item["y"]] for item in entries], dtype=float)
    else:
        df = pd.read_csv(path)
        pixels = df[["u", "v"]].to_numpy(float)
        world = df[["x", "y"]].to_numpy(float)
    if len(pixels) < 4:
        raise ValueError("Need at least 4 correspondences for homography")
    return pixels, world


def load_frame_index(path: Path | str) -> List[FrameRecord]:
    df = pd.read_csv(path)
    if "timestamp" not in df or "frame_id" not in df:
        raise ValueError("frame_index.csv must contain frame_id and timestamp columns")
    return [FrameRecord(int(r.frame_id), float(r.timestamp)) for r in df.itertuples()]


def load_ais(path: Path | str) -> List[AISTrack]:
    df = pd.read_csv(path)
    required = {"timestamp", "mmsi", "lon", "lat", "sog", "cog"}
    if not required.issubset(df.columns):
        raise ValueError(f"AIS csv missing columns: {required - set(df.columns)}")
    return [
        AISTrack(
            timestamp=float(r.timestamp),
            mmsi=int(r.mmsi),
            lon=float(r.lon),
            lat=float(r.lat),
            sog=float(r.sog),
            cog=float(r.cog),
        )
        for r in df.itertuples()
    ]


def interpolate_track(track: Sequence[AISTrack], timestamp: float) -> AISTrack | None:
    if not track:
        return None
    times = [t.timestamp for t in track]
    if timestamp < times[0] or timestamp > times[-1]:
        return None
    idx = np.searchsorted(times, timestamp)
    if idx == 0:
        return track[0]
    if idx == len(track):
        return track[-1]
    prev_pt = track[idx - 1]
    next_pt = track[idx]
    alpha = (timestamp - prev_pt.timestamp) / (next_pt.timestamp - prev_pt.timestamp + 1e-9)
    lon = (1 - alpha) * prev_pt.lon + alpha * next_pt.lon
    lat = (1 - alpha) * prev_pt.lat + alpha * next_pt.lat
    sog = (1 - alpha) * prev_pt.sog + alpha * next_pt.sog
    cog = (1 - alpha) * prev_pt.cog + alpha * next_pt.cog
    return AISTrack(timestamp=timestamp, mmsi=prev_pt.mmsi, lon=lon, lat=lat, sog=sog, cog=cog)


def geodetic_to_local_xy(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)
    x = EARTH_RADIUS * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS * (lat_rad - lat0_rad)
    return x, y


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    proj = (h @ homog.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def pack_prior_records(
    frames: Iterable[FrameRecord],
    track: Sequence[AISTrack],
    homography: np.ndarray,
    sigma_pixels: float,
    lon0: float,
    lat0: float,
    confidence: float,
) -> List[PriorRecord]:
    records: List[PriorRecord] = []
    world_track = []
    for item in track:
        x, y = geodetic_to_local_xy(item.lon, item.lat, lon0, lat0)
        world_track.append((x, y, item))
    xy = np.array([[p[0], p[1]] for p in world_track], dtype=float)
    proj_pixels = apply_homography(homography, xy)
    for idx, (frame, pixel) in enumerate(zip(frames, proj_pixels)):
        ais_state = track[idx]
        records.append(
            PriorRecord(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                mmsi=ais_state.mmsi,
                u=float(pixel[0]),
                v=float(pixel[1]),
                sigma_r=float(sigma_pixels),
                speed=float(ais_state.sog),
                course=float(ais_state.cog),
                confidence=float(confidence),
            )
        )
    return records


def save_json(path: Path | str, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_jsonl(path: Path | str, records: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for item in records:
            fp.write(json.dumps(item) + "\n")
