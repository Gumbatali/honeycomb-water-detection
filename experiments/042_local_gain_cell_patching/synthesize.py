#!/usr/bin/env python3
"""Paste independently permuted cells into neutral video with local +/-10% gain."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "honeycomb"
SOURCE_IMAGES = DATA / "images" / "train"
NEUTRAL_IMAGES = DATA / "synthetic" / "neutral_patch" / "neutral_water1" / "images"
PERMUTED = DATA / "synthetic" / "cell_permutation"
OUT = DATA / "synthetic" / "cell_permutation_gain"
GAINS = {
    1: (0.90, 0.94, 0.98, 1.02, 1.06, 1.10),
    2: (1.10, 1.02, 0.94, 1.06, 0.90, 0.98),
    3: (0.98, 1.10, 1.02, 0.90, 0.94, 1.06),
    4: (1.06, 0.90, 1.10, 0.94, 1.02, 0.98),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for index, gains in GAINS.items():
        final = OUT / f"gain_permutation_{index}"
        if final.exists(): print(f"exists {final.name}; skip"); continue
        partial = OUT / f".{final.name}.partial"; images = partial / "images"; masks = partial / "masks"
        images.mkdir(parents=True); masks.mkdir()
        source_variant = PERMUTED / f"permutation_{index}"
        metadata = json.loads((source_variant / "metadata.json").read_text())
        shifts = {int(cls): tuple(shift) for cls, shift in metadata["shifts_xy"].items()}
        target_mask = cv2.imread(str(source_variant / "masks" / "water1_frame_00050.png"), cv2.IMREAD_GRAYSCALE)
        height, width = target_mask.shape
        for frame_index in range(300):
            stem = f"water1_frame_{frame_index:05d}"
            source = np.load(SOURCE_IMAGES / f"{stem}.npy").astype(np.float32)
            source_neutral = np.load(NEUTRAL_IMAGES / f"{stem}.npy").astype(np.float32)
            destination_neutral = source_neutral.copy(); residual = source - source_neutral
            result = destination_neutral.copy()
            for cls, gain in enumerate(gains, 1):
                dx, dy = shifts[cls]; matrix = np.float32(((1, 0, dx), (0, 1, dy)))
                shifted_residual = cv2.warpAffine(residual, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                pixels = target_mask == cls
                result[pixels] = destination_neutral[pixels] + gain * shifted_residual[pixels]
            np.save(images / f"{stem}.npy", result.astype(np.float32))
            cv2.imwrite(str(masks / f"{stem}.png"), target_mask)
        output_metadata = {"source": "water1", "neutral_canvas": "neutral_water1", "frames": 300,
                           "cell_permutation": index, "class_gain": dict(enumerate(gains, 1)),
                           "formula": "neutral + gain * (defect - local_neutral)", "mask_transform_matches_patch": True}
        (partial / "metadata.json").write_text(json.dumps(output_metadata, indent=2) + "\n")
        (partial / "MANIFEST.txt").write_text(f"{final.name}\nframes: 300\nlocal gains: {gains}\nneutral_water1 retained as canvas\n")
        partial.rename(final); print(f"created {final.name}")


if __name__ == "__main__":
    main()
