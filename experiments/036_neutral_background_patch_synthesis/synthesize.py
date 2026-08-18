#!/usr/bin/env python3
"""Build neutral water1 and translated defect-patch videos with exact masks."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "honeycomb"
OUT = DATA / "synthetic" / "neutral_patch"
SOURCE = "water1"
SHAPE = (480, 640)
VARIANTS = {"patched_shift_left_down": (-46, 34), "patched_shift_right_up": (48, -30), "patched_shift_left_up": (-28, -42)}


def source_paths(frame: int) -> tuple[Path, Path]:
    stem = f"{SOURCE}_frame_{frame:05d}"
    return DATA / "images" / "train" / f"{stem}.npy", DATA / "masks" / "train" / f"{stem}.png"


def donor_maps(labels: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Map every class pixel to a fixed brighter local-background donor pixel."""
    reference = np.load(source_paths(50)[0]).astype(np.float32)
    all_defects = labels > 0; kernel = np.ones((35, 35), np.uint8); rng = np.random.default_rng(36017)
    maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cls in range(1, 7):
        target = labels == cls
        ring = (cv2.dilate(target.astype(np.uint8), kernel) > 0) & ~all_defects
        candidates = np.flatnonzero(ring)
        if not len(candidates): raise RuntimeError(f"No background donor pixels for class {cls}")
        bright_cut = np.quantile(reference.flat[candidates], .60)
        bright = candidates[reference.flat[candidates] >= bright_cut]
        destination = np.flatnonzero(target)
        donor = rng.choice(bright if len(bright) else candidates, len(destination), replace=True)
        maps[cls] = destination, donor
    return maps


def neutral_frame(frame: np.ndarray, donors: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    result = frame.copy().reshape(-1)
    original = frame.reshape(-1)
    for destination, donor in donors.values(): result[destination] = original[donor]
    return result.reshape(frame.shape)


def translated_labels(labels: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    dx, dy = shift; matrix = np.array(((1, 0, dx), (0, 1, dy)), dtype=np.float32)
    return cv2.warpAffine(labels, matrix, SHAPE[::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def transformed_cells(cells: list[dict], shift: tuple[int, int]) -> list[dict]:
    dx, dy = shift; output = []
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox_xyxy"]; updated = dict(cell)
        updated["bbox_xyxy"] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
        output.append(updated)
    return output


def build(video_id: str, labels: np.ndarray, donors: dict[int, tuple[np.ndarray, np.ndarray]], metadata: dict, shift: tuple[int, int] | None) -> None:
    final = OUT / video_id; partial = OUT / f".{video_id}.partial"
    if final.exists():
        print(f"exists {video_id}; skip"); return
    if partial.exists(): raise FileExistsError(f"Incomplete output exists: {partial}")
    images, masks = partial / "images", partial / "masks"; images.mkdir(parents=True); masks.mkdir()
    translated = translated_labels(labels, shift) if shift is not None else np.zeros_like(labels)
    matrix = np.array(((1, 0, shift[0]), (0, 1, shift[1])), dtype=np.float32) if shift is not None else None
    try:
        for index in range(3000):
            frame = np.load(source_paths(index)[0])
            neutral = neutral_frame(frame, donors)
            if matrix is None:
                result = neutral
            else:
                original = cv2.warpAffine(frame.astype(np.float32), matrix, SHAPE[::-1], flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(frame.dtype)
                result = neutral.copy(); result[translated > 0] = original[translated > 0]
            np.save(images / f"{SOURCE}_frame_{index:05d}.npy", result)
            cv2.imwrite(str(masks / f"{SOURCE}_frame_{index:05d}.png"), translated)
        manifest = {"video": video_id, "source_video": SOURCE, "frame_count": 3000, "image_shape": SHAPE,
                    "neutral_fill": "bright local background donor pixels", "shift_xy": shift,
                    "cells": transformed_cells(metadata["cells"], shift) if shift is not None else [],
                    "mask_values": {"0": "background", "1..6": "water20..water120"}}
        (partial / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (partial / "MANIFEST.txt").write_text(f"{video_id}\nsource: {SOURCE}\nshift_xy: {shift}\nframes: 3000\n"
                                              "labels: transformed exactly with copied defect patches\n")
        partial.rename(final); print(f"created {video_id}")
    except Exception:
        import shutil
        shutil.rmtree(partial, ignore_errors=True); raise


def main() -> None:
    labels = cv2.imread(str(source_paths(50)[1]), cv2.IMREAD_GRAYSCALE)
    metadata = json.loads((DATA / "metadata" / "videos" / f"{SOURCE}.json").read_text())
    donors = donor_maps(labels)
    build("neutral_water1", labels, donors, metadata, None)
    for name, shift in VARIANTS.items(): build(name, labels, donors, metadata, shift)


if __name__ == "__main__": main()
