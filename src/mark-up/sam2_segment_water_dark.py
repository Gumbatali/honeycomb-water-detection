#!/usr/bin/env python3
"""Segment the six dark cells in water1.mat with point-prompted SAM 2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from analyze_water_cooling import load_video_memmap


# Centres of the six dark rectangular cells in water1 (x, y), in source pixels.
DEFAULT_POINTS = "245,110;335,110;425,110;245,260;335,260;425,260"


def parse_points(value: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for item in value.split(";"):
        x, y = item.split(",")
        points.append((int(x), int(y)))
    return points


def scalar_to_rgb(image: np.ndarray) -> np.ndarray:
    """Turn one-channel thermal data into a contrast-enhanced RGB SAM input."""
    lo, hi = np.nanpercentile(image, (1, 99))
    gray = np.clip((image - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def thermal_rgb(video: np.memmap, frame_index: int, average: int) -> np.ndarray:
    start = max(0, frame_index - average // 2)
    stop = min(video.shape[2], start + average)
    image = np.mean(np.asarray(video[:, :, start:stop], dtype=np.float32), axis=2)
    return scalar_to_rgb(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2.1_hiera_small.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--frame", type=int, default=2700)
    parser.add_argument("--average", type=int, default=30, help="Number of adjacent frames to average")
    parser.add_argument("--input-map", type=Path, help="Optional H×W .npy map, e.g. cooling_delta.npy")
    parser.add_argument("--points", default=DEFAULT_POINTS, help="Positive points: x,y;x,y;...")
    parser.add_argument("--output", type=Path, default=Path("outputs/water1_sam2"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run: python -c \"import torch; print(torch.cuda.is_available())\" "
            "and fix the NVIDIA driver before running SAM 2."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    config_on_disk = Path("third_party/sam2/sam2") / args.config
    if not config_on_disk.is_file():
        raise FileNotFoundError(f"SAM 2 config not found: {config_on_disk}")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    video, _ = load_video_memmap(args.input, "data")
    if args.input_map:
        scalar_map = np.load(args.input_map)
        if scalar_map.shape != video.shape[:2]:
            raise ValueError(f"Expected map shape {video.shape[:2]}, got {scalar_map.shape}")
        image = scalar_to_rgb(scalar_map)
    else:
        if not 0 <= args.frame < video.shape[2]:
            raise ValueError(f"--frame must be from 0 to {video.shape[2] - 1}")
        image = thermal_rgb(video, args.frame, args.average)
    points = parse_points(args.points)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam2(args.config, str(args.checkpoint), device="cuda")
    predictor = SAM2ImagePredictor(model)
    masks: list[np.ndarray] = []
    scores: list[float] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        for point in points:
            predicted, confidence, _ = predictor.predict(
                point_coords=np.array([point], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
            best = int(np.argmax(confidence))
            masks.append(predicted[best])
            scores.append(float(confidence[best]))

    args.output.mkdir(parents=True, exist_ok=True)
    union = np.any(masks, axis=0)
    cv2.imwrite(str(args.output / "input_frame.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output / "dark_cells_mask.png"), (union * 255).astype(np.uint8))
    overlay = image.copy()
    overlay[union] = (0, 255, 80)
    for x, y in points:
        cv2.drawMarker(overlay, (x, y), (255, 40, 40), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    cv2.imwrite(str(args.output / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    (args.output / "result.json").write_text(
        json.dumps({"frame": args.frame, "average": args.average, "input_map": str(args.input_map) if args.input_map else None, "points_xy": points, "scores": scores}, indent=2) + "\n"
    )
    print(f"Saved SAM 2 masks to {args.output}")
    print("Scores:", ", ".join(f"{score:.3f}" for score in scores))


if __name__ == "__main__":
    main()
