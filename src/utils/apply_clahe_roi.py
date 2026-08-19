#!/usr/bin/env python3
"""Apply CLAHE only inside a rectangular ROI and keep the outside black."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def apply_clahe_roi(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    clip_limit: float,
    tile_size: int,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    if image.ndim != 2:
        raise ValueError("Expected a single-channel grayscale image")
    if not (0 <= x1 <= x2 < image.shape[1] and 0 <= y1 <= y2 < image.shape[0]):
        raise ValueError(f"ROI {roi} is outside image shape {image.shape}")

    result = np.zeros_like(image)
    crop = image[y1 : y2 + 1, x1 : x2 + 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    result[y1 : y2 + 1, x1 : x2 + 1] = clahe.apply(crop)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--x1", type=int, default=150)
    parser.add_argument("--y1", type=int, default=50)
    parser.add_argument("--x2", type=int, default=490)
    parser.add_argument("--y2", type=int, default=400)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-size", type=int, default=8)
    args = parser.parse_args()

    if args.clip_limit <= 0:
        raise ValueError("clip-limit must be positive")
    if args.tile_size < 2:
        raise ValueError("tile-size must be at least 2")

    image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(args.input)

    processed = apply_clahe_roi(
        image,
        (args.x1, args.y1, args.x2, args.y2),
        args.clip_limit,
        args.tile_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), processed):
        raise RuntimeError(f"Could not write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
