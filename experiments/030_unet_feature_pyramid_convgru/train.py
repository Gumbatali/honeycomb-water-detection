#!/usr/bin/env python3
"""Train frozen-U-Net-feature ConvGRU with video-level split and early stopping."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import UNetFeatureConvGRU


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "honeycomb"
SIZE = (192, 256)
THERMAL_FRAMES = np.arange(0, 300, 5, dtype=int)
UNET_FRAMES = np.array((10, 30, 50, 70, 100, 150, 200), dtype=int)
AUGMENTATIONS = ("011_horizontal_flip", "015_aggressive_rotation", "016_aggressive_affine")


class VideoDataset(Dataset):
    def __init__(self, records: list[tuple[str, str | None]], ignore_label: int | None = None,
                 thermal_normalization: str = "global", apply_roi: bool = False,
                 online_domain_augmentation: bool = False, online_temporal_augmentation: bool = False,
                 merge_label: int | None = None, merge_into: int | None = None) -> None:
        self.records, self.ignore_label = records, ignore_label
        self.thermal_normalization, self.apply_roi = thermal_normalization, apply_roi
        self.online_domain_augmentation = online_domain_augmentation
        self.online_temporal_augmentation = online_temporal_augmentation
        self.merge_label, self.merge_into = merge_label, merge_into
        if (merge_label is None) != (merge_into is None):
            raise ValueError("merge_label and merge_into must be specified together")
        self.augmentation_counts = {"samples": 0, "identity": 0, "rotation_180": 0, "scale_translation": 0,
                                    "cell_permutation": 0, "local_gain": 0, "temporal_shift": 0,
                                    "time_warp": 0, "cooling_warp": 0, "temporal_shift_seconds_sum": 0.0,
                                    "time_warp_sum": 0.0, "cooling_warp_sum": 0.0}

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video, augmentation = self.records[index]
        rotation_180 = scale_translation = local_gain = False
        scale, shift_x, shift_y = 1.0, 0.0, 0.0
        gain_by_class = np.ones(7, dtype=np.float32)
        temporal_shift, time_scale, cooling_scale = 0.0, 1.0, 1.0
        if self.online_domain_augmentation or self.online_temporal_augmentation:
            self.augmentation_counts["samples"] += 1
        if self.online_domain_augmentation:
            rotation_180 = np.random.random() < 0.5
            self.augmentation_counts["rotation_180" if rotation_180 else "identity"] += 1
            if np.random.random() < 0.5:
                augmentation = f"cell_permutation:permutation_{np.random.randint(1, 5)}"
                self.augmentation_counts["cell_permutation"] += 1
            scale_translation = np.random.random() < 0.7
            if scale_translation:
                scale = float(np.random.uniform(0.95, 1.05))
                shift_x, shift_y = float(np.random.uniform(-0.03, 0.03)), float(np.random.uniform(-0.03, 0.03))
                self.augmentation_counts["scale_translation"] += 1
            local_gain = np.random.random() < 0.7
            if local_gain:
                gain_by_class[1:] = np.random.uniform(0.9, 1.1, size=6)
                self.augmentation_counts["local_gain"] += 1
        if self.online_temporal_augmentation:
            if np.random.random() < 0.5:
                temporal_shift = float(np.random.uniform(-0.5, 0.5))
                self.augmentation_counts["temporal_shift"] += 1
                self.augmentation_counts["temporal_shift_seconds_sum"] += temporal_shift
            if np.random.random() < 0.5:
                time_scale = float(np.random.uniform(0.9, 1.1))
                self.augmentation_counts["time_warp"] += 1
                self.augmentation_counts["time_warp_sum"] += time_scale
            if np.random.random() < 0.3:
                cooling_scale = float(np.random.uniform(0.85, 1.15))
                self.augmentation_counts["cooling_warp"] += 1
                self.augmentation_counts["cooling_warp_sum"] += cooling_scale
        if augmentation is None:
            image_dir, mask_dir = DATA / "images" / "train", DATA / "masks" / "train"
        elif augmentation.startswith("neutral_patch:"):
            root = DATA / "synthetic" / "neutral_patch" / augmentation.split(":", 1)[1]
            image_dir, mask_dir = root / "images", root / "masks"
        elif augmentation.startswith("cell_permutation:"):
            root = DATA / "synthetic" / "cell_permutation" / augmentation.split(":", 1)[1]
            image_dir, mask_dir = root / "images", root / "masks"
        elif augmentation.startswith("cell_permutation_gain:"):
            root = DATA / "synthetic" / "cell_permutation_gain" / augmentation.split(":", 1)[1]
            image_dir, mask_dir = root / "images", root / "masks"
        else:
            root = DATA / "synthetic" / "materialized" / augmentation / video
            image_dir, mask_dir = root / "images", root / "masks"
        full_target = cv2.imread(str(mask_dir / f"{video}_frame_00050.png"), cv2.IMREAD_GRAYSCALE)

        def apply_local_gain(frames: np.ndarray, indices: np.ndarray) -> np.ndarray:
            if not local_gain: return frames
            neutral_dir = DATA / "synthetic" / "neutral_patch" / "neutral_water1" / "images"
            neutral = np.stack([np.load(neutral_dir / f"water1_frame_{int(i):05d}.npy").astype(np.float32) for i in indices])
            output = frames.copy()
            for cls in range(1, 7):
                pixels = full_target == cls
                output[:, pixels] = neutral[:, pixels] + gain_by_class[cls] * (frames[:, pixels] - neutral[:, pixels])
            return output

        def temporal_resample(corrected: np.ndarray, step_seconds: float) -> np.ndarray:
            if not self.online_temporal_augmentation: return corrected
            output_times = np.arange(len(corrected), dtype=np.float32) * step_seconds
            active_time = output_times - temporal_shift
            source_time = np.maximum(active_time, 0.0) * time_scale
            after_peak = active_time > 3.7
            source_time[after_peak] = (3.7 * time_scale +
                                       (active_time[after_peak] - 3.7) * time_scale * cooling_scale)
            positions = source_time / step_seconds
            low = np.floor(positions).astype(int); high = np.minimum(low + 1, len(corrected) - 1)
            low = np.clip(low, 0, len(corrected) - 1); fraction = positions - np.floor(positions)
            output = np.empty_like(corrected)
            for out_index, (lo, hi, weight) in enumerate(zip(low, high, fraction)):
                output[out_index] = corrected[lo] * (1.0 - weight) + corrected[hi] * weight
            output[active_time < 0] = 0
            return output

        # One shared radiometric scale for the entire 30-second temporal input.
        thermal = np.stack([np.load(image_dir / f"{video}_frame_{i:05d}.npy").astype(np.float32) for i in THERMAL_FRAMES])
        thermal = apply_local_gain(thermal, THERMAL_FRAMES)
        thermal = np.maximum(thermal - thermal[0], 0.0)
        thermal = temporal_resample(thermal, 0.5)
        if self.thermal_normalization == "pixel_peak":
            peak = thermal.max(axis=0); thermal = np.divide(thermal, peak[None], out=np.zeros_like(thermal), where=peak[None] > 1e-6)
        else:
            thermal /= max(float(thermal.max()), 1e-6)
        thermal = np.stack([cv2.resize(frame, SIZE[::-1], interpolation=cv2.INTER_LINEAR) for frame in thermal])[:, None]
        # Exact documented U-Net preparation: separate 0..20-second global scale.
        early = np.stack([np.load(image_dir / f"{video}_frame_{i:05d}.npy").astype(np.float32) for i in range(201)])
        early = apply_local_gain(early, np.arange(201))
        early = np.maximum(early - early[0], 0.0); early = temporal_resample(early, 0.1)
        early /= max(float(early.max()), 1e-6)
        roi = None
        if self.apply_roi:
            _, threshold = cv2.threshold(np.clip(early[50] * 255, 0, 255).astype(np.uint8), 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi = ndimage.binary_closing(threshold > 0, iterations=5)
            roi = ndimage.binary_fill_holes(roi); roi = ndimage.binary_opening(roi, iterations=3)
            components, count = ndimage.label(roi)
            if count: roi = components == (np.bincount(components.ravel())[1:].argmax() + 1)
            roi = ndimage.binary_erosion(roi, iterations=12)
            roi_small = cv2.resize(roi.astype(np.uint8), SIZE[::-1], interpolation=cv2.INTER_NEAREST) > 0
            thermal *= roi_small[None, None]
        unet_input = early[UNET_FRAMES]
        if roi is not None: unet_input *= roi[None]
        target = cv2.resize(full_target, SIZE[::-1], interpolation=cv2.INTER_NEAREST)
        if roi is not None: target[~roi_small] = 255
        if self.merge_label is not None:
            target[target == self.merge_label] = self.merge_into
        if self.ignore_label is not None:
            target[target == self.ignore_label] = 255
        if self.online_domain_augmentation:
            angle = 180.0 if rotation_180 else 0.0
            # water2 is a ~180-degree, 0.98-scale view with the rotated panel shifted upward by ~14.5%.
            base_scale = 0.98 if rotation_180 else 1.0
            base_y = -0.145 if rotation_180 else 0.0

            def matrix(width: int, height: int) -> np.ndarray:
                transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, base_scale * scale)
                transform[:, 2] += (shift_x * width, (base_y + shift_y) * height)
                return transform

            small_matrix = matrix(SIZE[1], SIZE[0]); full_matrix = matrix(unet_input.shape[2], unet_input.shape[1])
            thermal = np.stack([cv2.warpAffine(frame[0], small_matrix, SIZE[::-1], flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) for frame in thermal])[:, None]
            unet_input = np.stack([cv2.warpAffine(frame, full_matrix, (unet_input.shape[2], unet_input.shape[1]),
                                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                   for frame in unet_input])
            target = cv2.warpAffine(target, small_matrix, SIZE[::-1], flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return torch.from_numpy(thermal), torch.from_numpy(unet_input), torch.from_numpy(target.astype(np.int64))


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    classes = logits.shape[1]; valid = target != 255; safe = target.clone(); safe[~valid] = 0
    probability = logits.softmax(1)[:, 1:] * valid[:, None]
    onehot = F.one_hot(safe, classes).permute(0, 3, 1, 2).float()[:, 1:] * valid[:, None]
    numerator = 2 * (probability * onehot).sum((0, 2, 3)) + 1
    denominator = probability.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1
    return 1 - (numerator / denominator).mean()


def loss_parts(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    dice = dice_loss(logits, target)
    return ce + 0.5 * dice, ce, dice


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, weights: torch.Tensor, num_classes: int) -> dict[str, float]:
    model.eval(); inter = np.zeros(num_classes - 1); union = np.zeros(num_classes - 1); true_sum = np.zeros(num_classes - 1)
    predicted = np.zeros(num_classes); abs_error = total_water = valid_pixels = 0
    loss_sum = ce_sum = dice_sum = 0.0; batches = 0
    for thermal, unet_input, target in loader:
        target_device = target.to(device); logits = model(thermal.to(device), unet_input.to(device))
        loss, ce, dice_loss_value = loss_parts(logits, target_device, weights)
        loss_sum += loss.item(); ce_sum += ce.item(); dice_sum += dice_loss_value.item(); batches += 1
        prediction = logits.argmax(1).cpu().numpy(); truth = target.numpy(); valid = truth != 255
        water = (truth > 0) & valid; abs_error += np.abs(prediction[water] - truth[water]).sum(); total_water += water.sum()
        valid_pixels += valid.sum()
        for cls in range(num_classes): predicted[cls] += np.sum((prediction == cls) & valid)
        for cls in range(1, num_classes):
            inter[cls - 1] += np.sum((prediction == cls) & (truth == cls) & valid)
            union[cls - 1] += np.sum(((prediction == cls) | (truth == cls)) & valid)
            true_sum[cls - 1] += np.sum((truth == cls) & valid)
    iou = inter / np.maximum(union, 1); dice = 2 * inter / np.maximum(union + true_sum, 1)
    result = {"val_loss": loss_sum / max(batches, 1), "val_ce": ce_sum / max(batches, 1),
              "val_dice_loss": dice_sum / max(batches, 1), "macro_iou": float(iou.mean()),
              "macro_dice": float(dice.mean()), "ordinal_mae": float(abs_error / max(total_water, 1))}
    for cls in range(1, num_classes): result[f"iou_class_{cls}"] = float(iou[cls - 1])
    for cls in range(num_classes): result[f"pred_fraction_{cls}"] = float(predicted[cls] / max(valid_pixels, 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=96); parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-video", default="water1"); parser.add_argument("--valid-video", default="water2")
    parser.add_argument("--train-augmentation", action="append", default=list(AUGMENTATIONS))
    parser.add_argument("--include-patched-only", action="store_true")
    parser.add_argument("--include-cell-permutations", action="store_true")
    parser.add_argument("--include-cell-permutation-gain", action="store_true")
    parser.add_argument("--num-classes", type=int, default=7); parser.add_argument("--ignore-label", type=int)
    parser.add_argument("--merge-label", type=int); parser.add_argument("--merge-into", type=int)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--thermal-normalization", choices=("global", "pixel_peak"), default="global")
    parser.add_argument("--apply-roi", action="store_true")
    parser.add_argument("--online-domain-augmentation", action="store_true")
    parser.add_argument("--online-temporal-augmentation", action="store_true")
    parser.add_argument("--online-repeats", type=int, default=16)
    parser.add_argument("--unet-checkpoint", type=Path, default=ROOT / "models" / "segmentation" / "v1" / "unet_water_v2.pth")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    use_online = args.online_domain_augmentation or args.online_temporal_augmentation
    train_records = ([(args.train_video, None)] * args.online_repeats if use_online else
                     [(args.train_video, aug) for aug in (None, *args.train_augmentation)])
    if args.include_patched_only:
        if args.train_video != "water1": raise ValueError("patched videos currently derive from water1")
        names = ("patched_shift_left_down", "patched_shift_right_up", "patched_shift_left_up")
        train_records.extend(("water1", f"neutral_patch:{name}") for name in names)
    if args.include_cell_permutations:
        if args.train_video != "water1": raise ValueError("cell permutations currently derive from water1")
        train_records.extend(("water1", f"cell_permutation:permutation_{index}") for index in range(1, 5))
    if args.include_cell_permutation_gain:
        if args.train_video != "water1": raise ValueError("gain permutations currently derive from water1")
        train_records.extend(("water1", f"cell_permutation_gain:gain_permutation_{index}") for index in range(1, 5))
    dataset_options = {"ignore_label": args.ignore_label, "thermal_normalization": args.thermal_normalization,
                       "apply_roi": args.apply_roi, "merge_label": args.merge_label, "merge_into": args.merge_into}
    train_dataset = VideoDataset(train_records, **dataset_options, online_domain_augmentation=args.online_domain_augmentation,
                                 online_temporal_augmentation=args.online_temporal_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(VideoDataset([(args.valid_video, None)], **dataset_options), batch_size=1)
    torch.set_float32_matmul_precision("high")
    model = UNetFeatureConvGRU(args.hidden, args.dropout, args.num_classes).to(device)
    model.unet.load_checkpoint(torch.load(args.unet_checkpoint, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - 5, 1), eta_min=args.lr * 0.05)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, (warmup, cosine), milestones=(5,))
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    weights = torch.tensor((0.08,) + (1,) * (args.num_classes - 1), device=device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    best = {"macro_iou": -1.0}; stale = 0
    epoch_zero = evaluate(model, valid_loader, device, weights, args.num_classes); epoch_zero.update(epoch=0, lr=optimizer.param_groups[0]["lr"])
    history = [epoch_zero]
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = train_ce = train_dice = grad_norm_sum = 0.0; train_batches = 0
        for thermal, unet_input, target in train_loader:
            target = target.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(thermal.to(device), unet_input.to(device))
                loss, ce, dice = loss_parts(logits, target, weights)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm with {args.amp_dtype}; aborting instead of silently skipping updates")
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item(); train_ce += ce.item(); train_dice += dice.item(); grad_norm_sum += float(grad_norm); train_batches += 1
        scheduler.step()
        score = evaluate(model, valid_loader, device, weights, args.num_classes)
        score.update(train_loss=train_loss / train_batches, train_ce=train_ce / train_batches,
                     train_dice_loss=train_dice / train_batches, grad_norm=grad_norm_sum / train_batches,
                     epoch=epoch, lr=optimizer.param_groups[0]["lr"]); history.append(score)
        if score["macro_iou"] > best["macro_iou"] + 0.002:
            best = score.copy(); stale = 0
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                        "args": vars(args), "best": best, "epoch": epoch}, output / "best.pt")
        else: stale += 1
        if epoch == 1 or epoch % 5 == 0: print(f"epoch={epoch:03d} train_loss={score['train_loss']:.4f} val_loss={score['val_loss']:.4f} iou={score['macro_iou']:.4f} stale={stale}", flush=True)
        if stale >= args.patience:
            print(f"early_stop epoch={epoch}"); break
    report = {"architecture": "frozen U-Net e2/e3/e4 feature pyramid + ConvGRU", "best_validation": best,
              "epochs_completed": len(history) - 1, "train_sequences": len(train_records), "history": history,
              "online_augmentation_counts": train_dataset.augmentation_counts,
              "hyperparameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.txt").write_text(json.dumps(report["best_validation"], indent=2) + "\n")


if __name__ == "__main__": main()
