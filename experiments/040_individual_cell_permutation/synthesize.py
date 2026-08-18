#!/usr/bin/env python3
"""Create compact 30-second videos with water1 cells independently permuted."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "honeycomb"
OUT = DATA / "synthetic" / "cell_permutation"
SOURCE = "water1"
PERMUTATIONS = {
    "permutation_1": (3, 5, 1, 6, 2, 4),
    "permutation_2": (5, 1, 4, 2, 6, 3),
    "permutation_3": (2, 4, 6, 1, 3, 5),
    "permutation_4": (6, 3, 2, 5, 4, 1),
}
sys.path.insert(0, str(ROOT / "experiments" / "036_neutral_background_patch_synthesis"))
from synthesize import donor_maps, neutral_frame, source_paths  # noqa: E402


def main() -> None:
    labels = cv2.imread(str(source_paths(50)[1]), cv2.IMREAD_GRAYSCALE)
    donors = donor_maps(labels); height, width = labels.shape
    centers = {cls: np.argwhere(labels == cls).mean(0) for cls in range(1, 7)}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, slots in PERMUTATIONS.items():
        final = OUT / name
        if final.exists():
            print(f"exists {name}; skip"); continue
        partial = OUT / f".{name}.partial"; images = partial / "images"; masks = partial / "masks"
        images.mkdir(parents=True); masks.mkdir()
        shifts = {}
        transformed = {}
        combined = np.zeros_like(labels)
        for cls, slot in enumerate(slots, 1):
            dy, dx = np.rint(centers[slot] - centers[cls]).astype(int)
            shifts[cls] = (int(dx), int(dy)); matrix = np.float32(((1, 0, dx), (0, 1, dy)))
            transformed[cls] = cv2.warpAffine((labels == cls).astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST) > 0
            combined[transformed[cls]] = cls
        for frame_index in range(300):
            frame = np.load(source_paths(frame_index)[0]); result = neutral_frame(frame, donors)
            for cls in range(1, 7):
                dx, dy = shifts[cls]; matrix = np.float32(((1, 0, dx), (0, 1, dy)))
                shifted = cv2.warpAffine(frame.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(frame.dtype)
                result[transformed[cls]] = shifted[transformed[cls]]
            np.save(images / f"{SOURCE}_frame_{frame_index:05d}.npy", result)
            cv2.imwrite(str(masks / f"{SOURCE}_frame_{frame_index:05d}.png"), combined)
        metadata = {"source": SOURCE, "frames": 300, "class_to_destination_slot": dict(enumerate(slots, 1)),
                    "shifts_xy": shifts, "labels_transformed_with_pixels": True}
        (partial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (partial / "MANIFEST.txt").write_text(f"{name}\nsource: water1\nframes: 300\nindependent cell permutation: {slots}\n")
        partial.rename(final); print(f"created {name}")


if __name__ == "__main__":
    main()
