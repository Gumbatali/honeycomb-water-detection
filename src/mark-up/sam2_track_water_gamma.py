#!/usr/bin/env python3
"""Track the six dark water cells through the complete gamma-corrected video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

from  import open_thermal_data


DEFAULT_POINTS = "245,110;335,110;425,110;245,260;335,260;425,260"
DEFAULT_ROI = (150, 50, 490, 400)


def parse_points(value: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for item in value.split(";"):
        x, y = item.split(",")
        points.append((int(x), int(y)))
    return points


def scale_to_u8(image: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("scale_to_u8 expects a single-channel image")
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("image has no finite pixels")
    if low is None or high is None:
        low, high = np.percentile(finite, [5, 95])
    scale = max(float(high) - float(low), 1e-6)
    normalized = np.clip((image - float(low)) / scale, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def save_float_tiff(path: Path, image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError("save_float_tiff expects a single-channel image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.float32), mode="F").save(path, format="TIFF")


def dump_raw_video_frames(input_path: Path, output: Path, every: int = 10) -> None:
    if every < 1:
        raise ValueError("every must be at least 1")
    thermal = open_thermal_data(input_path)
    height, width, source_count = thermal.shape
    raw_dir = output / "raw_video_frames"
    raw_preview_dir = output / "raw_video_frames_preview"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_preview_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(0, source_count, every):
        target = raw_dir / f"{frame_index:05d}.tiff"
        if target.is_file():
            continue
        frame = np.asarray(thermal[:, :, frame_index], dtype=np.float32)
        save_float_tiff(target, frame)
        preview = scale_to_u8(frame)
        cv2.imwrite(str(raw_preview_dir / f"{frame_index:05d}.png"), preview)
        print(f"Raw dump: frame {frame_index}/{source_count - 1}")
    (output / "raw_video_dump.json").write_text(json.dumps({
        "input": str(input_path),
        "output_dir": str(raw_dir),
        "preview_dir": str(raw_preview_dir),
        "frames": len(range(0, source_count, every)),
        "source_frames": source_count,
        "every": every,
        "shape_hw": [height, width],
        "format": "float32_tiff",
    }, indent=2) + "\n")
    print(f"Saved raw video frames to {raw_dir}")


def extract_crop(
    img: np.ndarray,
    bbox: tuple[int, int, int, int] = DEFAULT_ROI,
):
    if img.ndim != 2:
        raise ValueError("extract_crop expects a single-channel image")

    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= x2 < img.shape[1] and 0 <= y1 <= y2 < img.shape[0]):
        raise ValueError(f"ROI {bbox} is outside image shape {img.shape}")

    return np.asarray(img[y1 : y2 + 1, x1 : x2 + 1], dtype=np.float32)


def preprocess_crop(
    crop: np.ndarray,
    gamma: float = 1.25,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    """Preprocess a crop and return an 8-bit single-channel image."""
    if crop.ndim != 2:
        raise ValueError("preprocess_crop expects a single-channel image")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        raise ValueError("crop has no finite pixels")

    if low is None or high is None:
        low, high = np.percentile(finite, [5, 95])
    scale = max(float(high) - float(low), 1e-6)
    normalized = np.clip((crop - float(low)) / scale, 0.0, 1.0)
    gamma_corrected = np.power(normalized, gamma)
    crop_u8 = (gamma_corrected * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(crop_u8)


def preprocess(
    img: np.ndarray,
    bbox: tuple[int, int, int, int] = DEFAULT_ROI,
    gamma: float = 1.25,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    """Preprocess and return only the ROI crop."""
    return preprocess_crop(extract_crop(img, bbox), gamma, low, high)


def paste_crop(
    crop: np.ndarray,
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    result = np.zeros(shape, dtype=crop.dtype)
    result[y1 : y2 + 1, x1 : x2 + 1] = crop
    return result


def roi_gamma_frame(frame: np.ndarray, gamma: float, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Normalize and gamma-transform only the selected ROI; everything else is zero."""
    return paste_crop(preprocess(frame, roi, gamma), frame.shape, roi)


def make_gamma_frames(input_path: Path, frame_dir: Path, gamma: float, stride: int, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Apply ROI-only gamma correction and keep every ``stride``-th source frame."""
    thermal = open_thermal_data(input_path)
    height, width, source_count = thermal.shape
    source_indices = range(0, source_count, stride)
    count = len(source_indices)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for sampled_index, source_index in enumerate(source_indices):
        target = frame_dir / f"{sampled_index:05d}.jpg"
        if target.is_file():
            continue
        frame = np.asarray(thermal[:, :, source_index], dtype=np.float32)
        corrected = roi_gamma_frame(frame, gamma, roi)
        if not cv2.imwrite(str(target), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {target}")
        if (sampled_index + 1) % 25 == 0:
            print(f"Gamma/ROI frames: {sampled_index + 1}/{count} (source frame {source_index})")
    return height, width, source_count, count


def gamma_frame(input_path: Path, frame_index: int, gamma: float, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Return one frame with the same gamma transformation used for the full video."""
    thermal = open_thermal_data(input_path)
    if not 0 <= frame_index < thermal.shape[2]:
        raise ValueError(f"preview frame must be within 0..{thermal.shape[2] - 1}")
    frame = np.asarray(thermal[:, :, frame_index], dtype=np.float32)
    return roi_gamma_frame(frame, gamma, roi)


def annotate_sampled_frames(
    input_path: Path,
    output: Path,
    checkpoint: Path,
    config: str,
    gamma: float,
    roi: tuple[int, int, int, int],
    points: list[tuple[int, int]],
    sample_every: int,
    sample_stop_frame: int | None,
) -> None:
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_on_disk = Path("third_party/sam2/sam2") / config
    if not config_on_disk.is_file():
        raise FileNotFoundError(config_on_disk)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    thermal = open_thermal_data(input_path)
    height, width, source_count = thermal.shape
    stop_frame = source_count // 2 if sample_stop_frame is None else sample_stop_frame
    if not 0 <= stop_frame < source_count:
        raise ValueError(f"sample_stop_frame must be within 0..{source_count - 1}")
    if not (0 <= roi[0] <= roi[2] < width and 0 <= roi[1] <= roi[3] < height):
        raise ValueError(f"ROI {roi} is outside video shape {(height, width)}")
    x1, y1, x2, y2 = roi
    local_points: list[tuple[int, int]] = []
    for x, y in points:
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            raise ValueError(f"Point {(x, y)} is outside ROI {roi}")
        local_points.append((x - x1, y - y1))

    frames_dir = output / "sampled_frames"
    raw_frames_dir = output / "sampled_raw_frames"
    raw_crops_dir = output / "sampled_raw_crops"
    raw_tiff_frames_dir = output / "sampled_raw_tiff_frames"
    raw_tiff_crops_dir = output / "sampled_raw_tiff_crops"
    raw_values_dir = output / "sampled_raw_values"
    raw_crop_values_dir = output / "sampled_raw_crop_values"
    masks_dir = output / "sampled_masks"
    overlays_dir = output / "sampled_overlays"
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    raw_crops_dir.mkdir(parents=True, exist_ok=True)
    raw_tiff_frames_dir.mkdir(parents=True, exist_ok=True)
    raw_tiff_crops_dir.mkdir(parents=True, exist_ok=True)
    raw_values_dir.mkdir(parents=True, exist_ok=True)
    raw_crop_values_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = build_sam2(config, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)

    records: list[dict[str, object]] = []
    frame_indices = list(range(0, stop_frame + 1, sample_every))

    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.inference_mode(), autocast_context:
        for n, frame_index in enumerate(frame_indices, 1):
            frame = np.asarray(thermal[:, :, frame_index], dtype=np.float32)
            crop = extract_crop(frame, roi)
            crop_preprocessed = preprocess_crop(crop, gamma)
            frame_visual = scale_to_u8(frame)
            raw_crop_visual = scale_to_u8(crop)
            crop_image = cv2.cvtColor(crop_preprocessed, cv2.COLOR_GRAY2RGB)
            predictor.set_image(crop_image)
            point_masks: list[np.ndarray] = []
            point_scores: list[float] = []
            for point in local_points:
                predicted, confidence, _ = predictor.predict(
                    point_coords=np.array([point], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True,
                )
                best = int(np.argmax(confidence))
                point_masks.append(predicted[best] > 0)
                point_scores.append(float(confidence[best]))
            crop_mask = np.any(point_masks, axis=0)
            mask = paste_crop(crop_mask.astype(np.uint8), (height, width), roi).astype(bool)
            score = float(np.mean(point_scores))

            stem = f"{frame_index:05d}"
            cv2.imwrite(str(raw_frames_dir / f"{stem}.png"), frame_visual)
            cv2.imwrite(str(raw_crops_dir / f"{stem}.png"), raw_crop_visual)
            save_float_tiff(raw_tiff_frames_dir / f"{stem}.tiff", frame)
            save_float_tiff(raw_tiff_crops_dir / f"{stem}.tiff", crop)
            cv2.imwrite(str(frames_dir / f"{stem}.png"), crop_preprocessed)
            np.save(raw_values_dir / f"{stem}.npy", frame)
            np.save(raw_crop_values_dir / f"{stem}.npy", crop)
            cv2.imwrite(str(masks_dir / f"{stem}.png"), (mask * 255).astype(np.uint8))
            overlay = cv2.cvtColor(paste_crop(crop_preprocessed, (height, width), roi), cv2.COLOR_GRAY2RGB)
            overlay[mask] = (0, 255, 80)
            for x, y in points:
                cv2.drawMarker(
                    overlay,
                    (x, y),
                    (255, 40, 40),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2,
                )
            cv2.imwrite(str(overlays_dir / f"{stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            records.append({
                "frame": frame_index,
                "mean_score": score,
                "scores": point_scores,
                "area_pixels": int(mask.sum()),
                "raw_min": float(np.nanmin(frame)),
                "raw_max": float(np.nanmax(frame)),
                "raw_mean": float(np.nanmean(frame)),
                "raw_std": float(np.nanstd(frame)),
                "crop_min": float(np.nanmin(crop)),
                "crop_max": float(np.nanmax(crop)),
                "crop_mean": float(np.nanmean(crop)),
                "crop_std": float(np.nanstd(crop)),
            })
            if n % 10 == 0 or n == len(frame_indices):
                print(
                    f"SAM sampled annotation: {n}/{len(frame_indices)} "
                    f"(source frame {frame_index}, raw {np.nanmin(frame):.3f}..{np.nanmax(frame):.3f}, "
                    f"crop {np.nanmin(crop):.3f}..{np.nanmax(crop):.3f})"
                )

    (output / "sampled_sam.json").write_text(json.dumps({
        "input": str(input_path),
        "gamma": gamma,
        "roi_xyxy_inclusive": list(roi),
        "points_xy": points,
        "local_points_xy": local_points,
        "raw_frames_dir": str(raw_frames_dir),
        "raw_crops_dir": str(raw_crops_dir),
        "raw_tiff_frames_dir": str(raw_tiff_frames_dir),
        "raw_tiff_crops_dir": str(raw_tiff_crops_dir),
        "raw_values_dir": str(raw_values_dir),
        "raw_crop_values_dir": str(raw_crop_values_dir),
        "preprocessed_frames_dir": str(frames_dir),
        "sample_every": sample_every,
        "sample_stop_frame": stop_frame,
        "frames": records,
    }, indent=2) + "\n")
    print(f"Saved sampled SAM annotations to {output}")


def chunk_folder(frame_dir: Path, work_dir: Path, indices: list[int]) -> Path:
    """Expose a contiguous global-frame range as locally numbered JPEGs for SAM 2."""
    folder = work_dir / f"{indices[0]:05d}_{indices[-1]:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    for local_index, global_index in enumerate(indices):
        link = folder / f"{local_index:05d}.jpg"
        if not link.exists():
            os.symlink((frame_dir / f"{global_index:05d}.jpg").resolve(), link)
    return folder


def write_mask(mask_dir: Path, frame_index: int, mask_logits: torch.Tensor) -> np.ndarray:
    mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
    target = mask_dir / f"{frame_index:05d}.png"
    if not cv2.imwrite(str(target), mask * 255):
        raise RuntimeError(f"Could not write {target}")
    return mask


def render_overlay(frame_dir: Path, mask_dir: Path, output: Path, count: int, width: int, height: int, fps: float) -> None:
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open the output MP4 encoder")
    try:
        for index in range(count):
            image = cv2.imread(str(frame_dir / f"{index:05d}.jpg"), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_dir / f"{index:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise RuntimeError(f"Missing processed frame or mask at {index}")
            green = np.zeros_like(image)
            green[:, :, 1] = 255
            selected = mask.astype(bool)
            image[selected] = cv2.addWeighted(image, 0.40, green, 0.60, 0)[selected]
            writer.write(image)
    finally:
        writer.release()


def track_chunk(predictor, frame_dir: Path, work_dir: Path, indices: list[int], seed_local: int, seed_mask: np.ndarray, mask_dir: Path, reverse: bool) -> np.ndarray:
    folder = chunk_folder(frame_dir, work_dir, indices)
    state = predictor.init_state(
        video_path=str(folder), offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    predictor.add_new_mask(state, frame_idx=seed_local, obj_id=1, mask=seed_mask)
    boundary_mask: np.ndarray | None = None
    for local_index, _, masks in predictor.propagate_in_video(
        state, start_frame_idx=seed_local, reverse=reverse
    ):
        global_index = indices[local_index]
        mask = write_mask(mask_dir, global_index, masks)
        if local_index == (0 if reverse else len(indices) - 1):
            boundary_mask = mask
    del state
    torch.cuda.empty_cache()
    if boundary_mask is None:
        raise RuntimeError("SAM 2 produced no masks for a chunk")
    return boundary_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2.1_hiera_small.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed-frame", type=int, default=2850)
    parser.add_argument("--stride", type=int, default=10, help="Keep every Nth source frame")
    parser.add_argument("--x1", type=int, default=150)
    parser.add_argument("--y1", type=int, default=50)
    parser.add_argument("--x2", type=int, default=490)
    parser.add_argument("--y2", type=int, default=400)
    parser.add_argument("--preview-frame", type=int, help="Only save one gamma-corrected frame and exit")
    parser.add_argument("--seed-mask", type=Path, default=Path("outputs/water1_sam2_cooling_map/dark_cells_mask.png"))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/water1_roi_gamma05_stride10"))
    parser.add_argument("--sample-every", type=int, help="Run SAM image annotation on every Nth source frame and exit")
    parser.add_argument("--sample-stop-frame", type=int, help="Last source frame for --sample-every; defaults to video midpoint")
    parser.add_argument("--sample-points", default=DEFAULT_POINTS, help="Positive points: x,y;x,y;...")
    parser.add_argument("--dump-raw-video", action="store_true", help="Save every source frame as float32 TIFF and exit")
    parser.add_argument("--dump-every", type=int, default=10, help="Keep every Nth frame for --dump-raw-video")
    args = parser.parse_args()

    if args.gamma <= 0 or args.chunk_size < 2 or args.stride < 1:
        parser.error("gamma must be positive, stride must be at least 1, and chunk-size must be at least 2")
    if args.sample_every is not None and args.sample_every < 1:
        parser.error("sample-every must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)
    roi = (args.x1, args.y1, args.x2, args.y2)
    if args.preview_frame is not None:
        preview = gamma_frame(args.input, args.preview_frame, args.gamma, roi)
        preview_path = args.output / f"gamma_{args.gamma:g}_frame_{args.preview_frame:05d}.png"
        cv2.imwrite(str(preview_path), preview)
        print(f"Saved gamma preview to {preview_path}")
        return
    if args.dump_raw_video:
        dump_raw_video_frames(args.input, args.output, args.dump_every)
        return
    if args.sample_every is not None:
        annotate_sampled_frames(
            args.input,
            args.output,
            args.checkpoint,
            args.config,
            args.gamma,
            roi,
            parse_points(args.sample_points),
            args.sample_every,
            args.sample_stop_frame,
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-video SAM 2 tracking; restart the NVIDIA driver/system")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (Path("third_party/sam2/sam2") / args.config).is_file():
        raise FileNotFoundError(args.config)

    from sam2.build_sam import build_sam2_video_predictor
    frame_dir = args.output / f"roi_gamma_{args.gamma:g}_stride_{args.stride}_frames"
    mask_dir = args.output / "masks"
    work_dir = args.output / "chunks"
    mask_dir.mkdir(exist_ok=True)
    height, width, source_count, count = make_gamma_frames(args.input, frame_dir, args.gamma, args.stride, roi)
    if not (0 <= args.x1 <= args.x2 < width and 0 <= args.y1 <= args.y2 < height):
        raise ValueError(f"ROI {roi} is outside video shape {(height, width)}")
    if not 0 <= args.seed_frame < source_count:
        raise ValueError(f"seed frame must be within 0..{source_count - 1}")
    seed_frame = min(count - 1, args.seed_frame // args.stride)
    seed_mask = cv2.imread(str(args.seed_mask), cv2.IMREAD_GRAYSCALE)
    if seed_mask is None or seed_mask.shape != (height, width):
        raise ValueError(f"Seed mask must have shape {(height, width)}: {args.seed_mask}")
    seed_mask = (seed_mask > 0)
    roi_mask = np.zeros_like(seed_mask, dtype=bool)
    roi_mask[args.y1:args.y2 + 1, args.x1:args.x2 + 1] = True
    seed_mask &= roi_mask

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    # Propagate from the seed to the final frame, carrying the boundary mask into each chunk.
    start = seed_frame
    forward_seed = seed_mask
    while start < count - 1:
        end = min(start + args.chunk_size - 1, count - 1)
        indices = list(range(start, end + 1))
        print(f"Forward: {start}..{end}")
        forward_seed = track_chunk(predictor, frame_dir, work_dir, indices, 0, forward_seed, mask_dir, reverse=False)
        start = end

    # Then propagate from the same seed to frame 0 in reverse time order.
    end = seed_frame
    reverse_seed = seed_mask
    while end > 0:
        start = max(0, end - args.chunk_size + 1)
        indices = list(range(start, end + 1))
        print(f"Reverse: {start}..{end}")
        reverse_seed = track_chunk(predictor, frame_dir, work_dir, indices, len(indices) - 1, reverse_seed, mask_dir, reverse=True)
        end = start

    # Ensure the seed frame exists even for a one-frame video.
    if not (mask_dir / f"{seed_frame:05d}.png").is_file():
        cv2.imwrite(str(mask_dir / f"{seed_frame:05d}.png"), seed_mask.astype(np.uint8) * 255)
    render_overlay(frame_dir, mask_dir, args.output / f"overlay_gamma_{args.gamma:g}_stride_{args.stride}.mp4", count, width, height, args.fps / args.stride)
    (args.output / "run.json").write_text(json.dumps({
        "input": str(args.input), "gamma": args.gamma, "source_seed_frame": args.seed_frame,
        "sampled_seed_frame": seed_frame, "stride": args.stride,
        "roi_xyxy_inclusive": [args.x1, args.y1, args.x2, args.y2],
        "seed_mask": str(args.seed_mask), "source_frames": source_count,
        "sampled_frames": count, "chunk_size": args.chunk_size,
    }, indent=2) + "\n")
    print(f"Saved full-video masks and overlay to {args.output}")


if __name__ == "__main__":
    main()
