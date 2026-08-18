#!/usr/bin/env python3
"""Create reproducible, label-aware video augmentation experiments.

By default this writes virtual-video manifests and QA plots, not tens of GB of
duplicated frames.  Pass --materialize to write every transformed frame.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "honeycomb"
EXPERIMENTS = ROOT / "experiments"
SYNTHETIC = ROOT / "data" / "synthetic" / "video_augmentation_manifests"
VIDEOS = ("water1", "water2", "water4")
SHAPE = (480, 640)


@dataclass(frozen=True)
class Augmentation:
    experiment: str
    title: str
    kind: str
    matrix: np.ndarray | None
    params: dict[str, Any]
    explanation: str


def affine_matrix(angle: float = 0, scale: float = 1, shear_x: float = 0, dx: float = 0, dy: float = 0) -> np.ndarray:
    h, w = SHAPE
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 1] += shear_x
    matrix[:, 2] += (dx, dy)
    return matrix.astype(np.float32)


def definitions() -> tuple[Augmentation, ...]:
    h, w = SHAPE
    return (
        Augmentation("010_rotation", "Small rotation", "affine", affine_matrix(angle=6), {"angle_degrees": 6},
                     "Small camera/panel rotation; the same affine matrix is used for every frame and label."),
        Augmentation("011_horizontal_flip", "Horizontal flip", "affine", np.array([[-1, 0, w - 1], [0, 1, 0]], np.float32), {},
                     "Mirror view.  Class IDs are preserved, while every mask and bbox is mirrored."),
        Augmentation("012_geometric_affine", "Affine geometry", "affine", affine_matrix(angle=-4, scale=0.98, shear_x=0.025, dx=3, dy=-2),
                     {"angle_degrees": -4, "scale": 0.98, "shear_x": 0.025, "dx": 3, "dy": -2},
                     "Mild rotation, scale, shear and translation to model viewpoint and mounting variation."),
        Augmentation("013_background_patching", "Small background patches", "patch", None,
                     {"patch_count": 4, "patch_size": 12, "bias_celsius": 0.10, "seed": 113},
                     "Fixed small sensor-bias patches are placed only on background pixels; masks and boxes stay unchanged."),
        Augmentation("014_defect_location_shift", "Defect-location shift", "affine", affine_matrix(dx=8, dy=-6),
                     {"dx": 8, "dy": -6},
                     "Rigid panel shift moves the thermal defect regions and their labels together by less than a cell pitch."),
        Augmentation("015_aggressive_rotation", "Aggressive rotation", "affine", affine_matrix(angle=15), {"angle_degrees": 15},
                     "Large but still plausible ±15-degree camera/panel rotation for robust spatial invariance."),
        Augmentation("016_aggressive_affine", "Aggressive affine geometry", "affine",
                     affine_matrix(angle=-12, scale=0.90, shear_x=0.08, dx=12, dy=-8),
                     {"angle_degrees": -12, "scale": 0.90, "shear_x": 0.08, "dx": 12, "dy": -8},
                     "Stronger viewpoint variation: rotation, 10% scale change, shear and sub-cell translation."),
    )


def transform_mask(mask: np.ndarray, aug: Augmentation) -> np.ndarray:
    if aug.kind != "affine":
        return mask.copy()
    h, w = mask.shape
    return cv2.warpAffine(mask, aug.matrix, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def transform_frame(frame: np.ndarray, aug: Augmentation, patches: list[list[int]]) -> np.ndarray:
    if aug.kind == "affine":
        h, w = frame.shape
        # OpenCV 5 does not implement remap for float16, whereas source thermal
        # frames are float16.  Warp in float32 and preserve the storage dtype.
        warped = cv2.warpAffine(frame.astype(np.float32), aug.matrix, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        return warped.astype(frame.dtype)
    output = frame.astype(np.float32, copy=True)
    for x, y, size, bias in patches:
        output[y:y + size, x:x + size] += bias
    return output.astype(frame.dtype)


def bbox_after_affine(box: list[int], matrix: np.ndarray) -> list[int] | None:
    x0, y0, x1, y1 = box
    corners = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
    mapped = cv2.transform(corners, matrix)[0]
    h, w = SHAPE
    left, top = np.maximum(np.floor(mapped.min(axis=0)), (0, 0)).astype(int)
    right, bottom = np.minimum(np.ceil(mapped.max(axis=0)), (w, h)).astype(int)
    return [int(left), int(top), int(right), int(bottom)] if right > left and bottom > top else None


def choose_background_patches(mask: np.ndarray, params: dict[str, Any]) -> list[list[int]]:
    rng = np.random.default_rng(params["seed"])
    size, count = int(params["patch_size"]), int(params["patch_count"])
    h, w = mask.shape
    patches: list[list[int]] = []
    while len(patches) < count:
        x, y = int(rng.integers(0, w - size)), int(rng.integers(0, h - size))
        if not mask[y:y + size, x:x + size].any():
            sign = -1 if len(patches) % 2 else 1
            patches.append([x, y, size, sign * float(params["bias_celsius"])])
    return patches


def update_metadata(source: dict[str, Any], aug: Augmentation, patches: list[list[int]]) -> dict[str, Any]:
    result = {key: value for key, value in source.items() if key != "frames"}
    result["video"] = f"{source['video']}__{aug.experiment}"
    result["source_video"] = source["video"]
    result["augmentation"] = {"name": aug.experiment, "kind": aug.kind, "params": aug.params,
                              "matrix_2x3": aug.matrix.tolist() if aug.matrix is not None else None,
                              "patches_xy_size_bias": patches}
    cells = []
    for cell in source["cells"]:
        updated = dict(cell)
        if aug.matrix is not None:
            transformed = bbox_after_affine(cell["bbox_xyxy"], aug.matrix)
            if transformed is None:
                continue
            updated["bbox_xyxy"] = transformed
        cells.append(updated)
    result["cells"] = cells
    return result


def plot_qa(video: str, aug: Augmentation, source_mask: np.ndarray, output: Path, patches: list[list[int]]) -> None:
    frame_path = DATA / "images" / "train" / f"{video}_frame_00050.npy"
    frame = np.load(frame_path).astype(np.float32)
    transformed_frame = transform_frame(frame, aug, patches)
    transformed_mask = transform_mask(source_mask, aug)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, image, title in zip(axes.flat, (frame, transformed_frame, source_mask, transformed_mask),
                                  ("source thermal frame 50", aug.title, "source labels", "augmented labels")):
        axis.imshow(image, cmap="inferno" if image.dtype.kind == "f" else "tab10")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{aug.experiment}: {video}")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def materialize(video: str, aug: Augmentation, patches: list[list[int]], output_dir: Path) -> None:
    source = DATA / "images" / "train"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing materialized video: {output_dir}")
    temporary_dir = output_dir.with_name(f".{output_dir.name}.partial")
    if temporary_dir.exists():
        raise FileExistsError(f"Remove or inspect incomplete materialization first: {temporary_dir}")
    image_out, mask_out = temporary_dir / "images", temporary_dir / "masks"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    try:
        for frame_path in sorted(source.glob(f"{video}_frame_*.npy")):
            frame = np.load(frame_path)
            np.save(image_out / frame_path.name, transform_frame(frame, aug, patches))
            mask_path = DATA / "masks" / "train" / f"{frame_path.stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(mask_out / mask_path.name), transform_mask(mask, aug))
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def write_experiment_readme(aug: Augmentation) -> None:
    directory = EXPERIMENTS / aug.experiment
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "README.txt").write_text(
        f"Experiment {aug.experiment}: {aug.title}\n{'=' * (len(aug.experiment) + len(aug.title) + 13)}\n\n"
        f"{aug.explanation}\n\n"
        "Unit of augmentation: a whole video.  Parameters are deterministic and\n"
        "identical for every frame, preserving each cell's heating/cooling curve.\n"
        "Semantic masks use nearest-neighbour interpolation; thermal frames use\n"
        "linear interpolation.  Bounding boxes are calculated from transformed\n"
        "corners and saved in the manifest.\n\n"
        "Data product: data/synthetic/video_augmentation_manifests/<experiment>/\n"
        "contains one manifest per source video and plots/ contains QA diagrams.\n"
        "Full cached frames and masks, when requested, are under\n"
        "data/honeycomb/synthetic/materialized/.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true", help="write all transformed NPY/PNG frames (large)")
    parser.add_argument("--only", nargs="+", help="only these augmentation IDs")
    args = parser.parse_args()
    selected = [aug for aug in definitions() if args.only is None or aug.experiment in args.only]
    unknown = set(args.only or ()) - {aug.experiment for aug in definitions()}
    if unknown:
        raise ValueError(f"Unknown augmentation IDs: {sorted(unknown)}")
    for aug in selected:
        write_experiment_readme(aug)
        manifest_dir, plot_dir = SYNTHETIC / aug.experiment, EXPERIMENTS / aug.experiment / "plots"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for video in VIDEOS:
            metadata = json.loads((DATA / "metadata" / "videos" / f"{video}.json").read_text())
            mask = cv2.imread(str(DATA / "masks" / "train" / f"{video}_frame_00050.png"), cv2.IMREAD_GRAYSCALE)
            patches = choose_background_patches(mask, aug.params) if aug.kind == "patch" else []
            (manifest_dir / f"{video}.json").write_text(json.dumps(update_metadata(metadata, aug, patches), indent=2) + "\n")
            plot_qa(video, aug, mask, plot_dir / f"{video}.png", patches)
            if args.materialize:
                materialize(video, aug, patches, DATA / "synthetic" / "materialized" / aug.experiment / video)
        print(f"created {aug.experiment}")


if __name__ == "__main__":
    main()
