#!/usr/bin/env python3
"""Create initial annotations for a one-channel thermal MAT video.

SAM 2.1 does not accept a text prompt.  This script converts the semantic
request ("dark areas that cool faster") into point and box prompts that can be
fed to SAM 2 afterwards.  It also produces a reproducible, non-model-based
first-pass mask, which is useful for checking whether the requested phenomenon
is actually visible in the recording.

The reader intentionally uses numpy.memmap: water1.mat is several gigabytes
and loading the whole cube with scipy.io.loadmat would waste RAM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


MI_MATRIX = 14
MI_SINGLE = 7
MI_DOUBLE = 9


@dataclass(frozen=True)
class MatArray:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    offset: int


def _element(stream, offset: int, endian: str) -> tuple[int, int, int, int]:
    """Return Matlab element type, byte count, data offset, and next offset."""
    stream.seek(offset)
    tag = stream.read(8)
    if len(tag) != 8:
        raise ValueError("Unexpected end of MAT file")
    first, second = struct.unpack(f"{endian}II", tag)
    small_type, small_size = struct.unpack(f"{endian}HH", tag[:4])
    if small_size and small_size <= 4:
        return small_type, small_size, offset + 4, offset + 8
    next_offset = offset + 8 + ((second + 7) // 8) * 8
    return first, second, offset + 8, next_offset


def find_mat_arrays(path: Path) -> list[MatArray]:
    """Locate uncompressed numeric arrays in a MATLAB v5 file."""
    with path.open("rb") as stream:
        header = stream.read(128)
        if len(header) != 128 or header[126:128] not in (b"IM", b"MI"):
            raise ValueError("Expected an uncompressed MATLAB v5 MAT file")
        endian = "<" if header[126:128] == b"IM" else ">"
        file_size = path.stat().st_size
        cursor = 128
        arrays: list[MatArray] = []

        while cursor + 8 <= file_size:
            stream.seek(cursor)
            raw = stream.read(8)
            element_type, element_size = struct.unpack(f"{endian}II", raw)
            next_matrix = cursor + 8 + ((element_size + 7) // 8) * 8
            if element_type != MI_MATRIX:
                cursor = next_matrix
                continue

            inner = cursor + 8
            _, _, _, inner = _element(stream, inner, endian)  # array flags
            _, dims_size, dims_offset, inner = _element(stream, inner, endian)
            stream.seek(dims_offset)
            shape = tuple(struct.unpack(f"{endian}{dims_size // 4}i", stream.read(dims_size)))
            _, name_size, name_offset, inner = _element(stream, inner, endian)
            stream.seek(name_offset)
            name = stream.read(name_size).decode("ascii")
            data_type, data_size, data_offset, _ = _element(stream, inner, endian)
            dtype = {MI_SINGLE: np.dtype(f"{endian}f4"), MI_DOUBLE: np.dtype(f"{endian}f8")}.get(data_type)
            if dtype is not None and data_size == int(np.prod(shape)) * dtype.itemsize:
                arrays.append(MatArray(name=name, shape=shape, dtype=dtype, offset=data_offset))
            cursor = next_matrix
    return arrays


def load_video_memmap(path: Path, key: str) -> tuple[np.memmap, float]:
    arrays = find_mat_arrays(path)
    selected = next((array for array in arrays if array.name == key), None)
    if selected is None or len(selected.shape) != 3:
        available = ", ".join(f"{array.name}{array.shape}" for array in arrays)
        raise ValueError(f"No 3-D array named {key!r}; found: {available}")
    video = np.memmap(path, mode="r", dtype=selected.dtype, offset=selected.offset, shape=selected.shape, order="F")
    fps_array = next((array for array in arrays if array.name.lower() == "fps" and array.shape == (1, 1)), None)
    fps = 0.0
    if fps_array is not None:
        fps = float(np.memmap(path, mode="r", dtype=fps_array.dtype, offset=fps_array.offset, shape=(1,))[0])
    return video, fps


def mean_window(video: np.memmap, indices: range) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = video.shape
    total = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.uint32)
    for index in indices:
        frame = np.asarray(video[:, :, index], dtype=np.float32)
        finite = np.isfinite(frame)
        total += np.where(finite, frame, 0.0)
        count += finite
    return (total / np.maximum(count, 1)).astype(np.float32), count


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def component_prompts(mask: np.ndarray, score: np.ndarray, limit: int = 8) -> list[dict[str, object]]:
    labels, count = ndimage.label(mask)
    prompts: list[dict[str, object]] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if not len(xs):
            continue
        local_scores = score[ys, xs]
        best = int(np.argmax(local_scores))
        prompts.append(
            {
                "point_xy": [int(xs[best]), int(ys[best])],
                "label": 1,
                "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "area_pixels": int(len(xs)),
                "mean_score": float(local_scores.mean()),
            }
        )
    return sorted(prompts, key=lambda item: item["mean_score"], reverse=True)[:limit]


def write_preview(output: Path, image: np.ndarray, dark: np.ndarray, fast: np.ndarray) -> None:
    low, high = np.nanpercentile(image, (2, 98))
    gray = np.clip((image - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[dark] = (0, 220, 255)       # cyan: dark areas
    rgb[fast] = (255, 64, 48)       # red: faster cooling
    rgb[dark & fast] = (255, 230, 0)  # yellow: both criteria
    ppm = output.with_suffix(".ppm")
    with ppm.open("wb") as stream:
        stream.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode())
        stream.write(np.ascontiguousarray(rgb).tobytes())
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ppm), str(output)], check=True)
        ppm.unlink(missing_ok=True)


def write_overlay_video(video: np.memmap, output: Path, fps: float, stride: int, dark: np.ndarray, fast: np.ndarray) -> None:
    if not shutil.which("ffmpeg"):
        return
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{video.shape[1]}x{video.shape[0]}", "-framerate", str(fps / stride),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in range(0, video.shape[2], stride):
            frame = np.asarray(video[:, :, index], dtype=np.float32)
            low, high = np.nanpercentile(frame, (2, 98))
            gray = np.clip((frame - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            rgb[dark] = (0, 220, 255)
            rgb[fast] = (255, 64, 48)
            rgb[dark & fast] = (255, 230, 0)
            process.stdin.write(np.ascontiguousarray(rgb).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg could not encode the overlay video")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="MATLAB v5 file containing a H×W×T single-channel array")
    parser.add_argument("--key", default="data", help="Name of the 3-D MAT array (default: data)")
    parser.add_argument("--output", type=Path, default=Path("outputs/water1_initial_annotation"))
    parser.add_argument("--early-fraction", type=float, default=0.10, help="Fraction of early frames used as baseline")
    parser.add_argument("--late-fraction", type=float, default=0.10, help="Fraction of late frames used for comparison")
    parser.add_argument("--dark-quantile", type=float, default=0.20)
    parser.add_argument("--cooling-quantile", type=float, default=0.85)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--fps", type=float, default=None, help="FPS used only for overlay.mp4; defaults to MAT metadata or 30")
    parser.add_argument("--video-stride", type=int, default=10, help="Keep every Nth frame in the overlay MP4")
    args = parser.parse_args()

    if not 0 < args.early_fraction <= 1 or not 0 < args.late_fraction <= 1:
        parser.error("window fractions must be within (0, 1]")
    video, metadata_fps = load_video_memmap(args.input, args.key)
    fps = args.fps if args.fps is not None else (metadata_fps if metadata_fps > 0 else 30.0)
    frames = video.shape[2]
    early_count = max(1, round(frames * args.early_fraction))
    late_count = max(1, round(frames * args.late_fraction))
    early, early_valid = mean_window(video, range(early_count))
    late, late_valid = mean_window(video, range(frames - late_count, frames))
    valid = (early_valid > early_count * 0.95) & (late_valid > late_count * 0.95)
    cooling = early - late  # positive values mean the signal became colder/darker

    dark_threshold = float(np.nanquantile(late[valid], args.dark_quantile))
    cooling_threshold = float(np.nanquantile(cooling[valid], args.cooling_quantile))
    dark = remove_small_regions(valid & (late <= dark_threshold), args.min_area)
    fast = remove_small_regions(valid & (cooling >= cooling_threshold), args.min_area)
    candidate = dark & fast

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "early_mean.npy", early)
    np.save(args.output / "late_mean.npy", late)
    np.save(args.output / "cooling_delta.npy", cooling)
    np.save(args.output / "dark_mask.npy", dark)
    np.save(args.output / "fast_cooling_mask.npy", fast)
    np.save(args.output / "dark_and_fast_mask.npy", candidate)
    write_preview(args.output / "preview.png", late, dark, fast)
    write_overlay_video(video, args.output / "overlay.mp4", fps, args.video_stride, dark, fast)

    semantic_prompt = (
        "Одноканальная тепловая последовательность. Выдели связные области, которые на поздних кадрах "
        "темнее или холоднее локального фона и при этом демонстрируют устойчиво большее падение "
        "температуры (интенсивности) между начальным и конечным временными окнами. Игнорируй шум, "
        "границы кадра и единичные кратковременные затемнения."
    )
    metadata = {
        "input": str(args.input), "shape_hwt": list(video.shape), "fps": fps,
        "definition": {"dark": "late-frame lower quantile", "fast_cooling": "early_mean - late_mean"},
        "thresholds": {"dark_late_value": dark_threshold, "fast_cooling_delta": cooling_threshold},
        "semantic_prompt": semantic_prompt,
        "sam2_note": "SAM 2.1 uses point/box prompts rather than a text prompt. Use these positive seeds on frame 0 or a representative early frame.",
        "dark_area_prompts": component_prompts(dark, -late),
        "fast_cooling_prompts": component_prompts(fast, cooling),
        "dark_and_fast_prompts": component_prompts(candidate, cooling),
    }
    (args.output / "sam2_prompts.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Created initial annotations in: {args.output}")
    print(f"Video shape: {video.shape}; FPS: {fps:g}")
    print(f"Thresholds — dark <= {dark_threshold:.4g}; fast cooling >= {cooling_threshold:.4g}")
    print(f"Areas — dark: {dark.sum()} px; fast cooling: {fast.sum()} px; both: {candidate.sum()} px")


if __name__ == "__main__":
    main()
