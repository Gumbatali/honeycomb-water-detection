#!/usr/bin/env python3
"""Inference + metrics + mask visualisation for the temporal checkpoint.

Loads models/temporal/best.pt (experiment 062: frozen U-Net feature pyramid +
full-frame ConvGRU) and runs it over the three labelled panels water1/water2/
water4, reusing the exact pre/post-processing implemented in
experiments/030_unet_feature_pyramid_convgru/train.py (VideoDataset).

Outputs per video:
  <video>_overlay.png     thermal@5s with the predicted water mask overlaid
  <video>_panels.png      thermal / ground-truth / prediction / error
  <video>_confusion.json  per-pixel confusion matrix (deployed classes)
and a combined metrics.json + summary.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "030_unet_feature_pyramid_convgru"
sys.path.insert(0, str(SOURCE))

from model import UNetFeatureConvGRU, UNetFeatureConvGRUMultitask  # noqa: E402
from train import VideoDataset  # noqa: E402

VIDEOS = ("water1", "water2", "water4")
CLASS_NAMES = ("background", "water20", "water40", "water60/80", "water100")


def build_model(checkpoint: dict) -> UNetFeatureConvGRU:
    args = checkpoint["args"]
    model_class = UNetFeatureConvGRUMultitask if args.get("multitask_heads", False) else UNetFeatureConvGRU
    thermal_channels = 2 if args.get("thermal_representation", "absolute") == "both" else 1
    model = model_class(
        args["hidden"],
        args["dropout"],
        args["num_classes"],
        thermal_channels,
        args.get("separate_thermal_stems", False),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def dataset_options(args: dict) -> dict:
    return {
        "ignore_label": args.get("ignore_label"),
        "thermal_normalization": args.get("thermal_normalization", "global"),
        "apply_roi": args.get("apply_roi", False),
        "merge_label": args.get("merge_label"),
        "merge_into": args.get("merge_into"),
        "compact_after_merge": args.get("compact_after_merge", False),
        "neutralize_label": args.get("neutralize_label"),
        "thermal_representation": args.get("thermal_representation", "absolute"),
        "contrast_normalization": args.get("contrast_normalization", "pixel_peak"),
    }


def metrics(prediction: np.ndarray, truth: np.ndarray, num_classes: int) -> dict:
    valid = truth != 255
    prediction = prediction.copy()
    prediction[~valid] = 0  # deployment post-processing: no defect outside ROI

    iou, dice = [], []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t in range(num_classes):
        for p in range(num_classes):
            confusion[t, p] = int(np.sum((truth == t) & (prediction == p) & valid))
    for cls in range(1, num_classes):
        inter = int(np.sum((prediction == cls) & (truth == cls) & valid))
        union = int(np.sum(((prediction == cls) | (truth == cls)) & valid))
        denom = int(np.sum((truth == cls) & valid)) + int(np.sum((prediction == cls) & valid))
        iou.append(inter / max(union, 1))
        dice.append(2 * inter / max(denom, 1))
    correct = int(np.sum((prediction == truth) & valid))
    total = int(valid.sum())
    return {
        "macro_iou": float(np.mean(iou)),
        "macro_dice": float(np.mean(dice)),
        "per_class_iou": {CLASS_NAMES[i]: iou[i - 1] for i in range(1, num_classes)},
        "per_class_dice": {CLASS_NAMES[i]: dice[i - 1] for i in range(1, num_classes)},
        "pixel_accuracy": correct / max(total, 1),
        "valid_pixels": total,
        "confusion_matrix": confusion.tolist(),
    }


def render(video: str, thermal: torch.Tensor, truth: np.ndarray, prediction: np.ndarray, out_dir: Path) -> None:
    valid = truth != 255
    display_truth = truth.copy(); display_truth[~valid] = 0
    thermal_5s = thermal[10, 0].numpy()

    # Overlay: thermal under a translucent colour mask.
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.imshow(thermal_5s, cmap="inferno")
    masked = np.ma.masked_where(prediction == 0, prediction)
    ax.imshow(masked, cmap="tab10", vmin=0, vmax=4, alpha=0.55, interpolation="nearest")
    ax.set_title(f"{video}: thermal@5s + predicted water mask")
    ax.axis("off")
    fig.savefig(out_dir / f"{video}_overlay.png", dpi=160); plt.close(fig)

    error = ((prediction != truth) & valid).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    panels = (
        (thermal_5s, "thermal at 5 s", "inferno", None),
        (display_truth, "ground-truth mask", "tab10", 4),
        (prediction, "prediction", "tab10", 4),
        (error, "pixel error", "Reds", 1),
    )
    for axis, (image, title, cmap, vmax) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0 if vmax is not None else None, vmax=vmax)
        axis.set_title(title); axis.axis("off")
    fig.suptitle(video)
    fig.savefig(out_dir / f"{video}_panels.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "temporal" / "best.pt")
    parser.add_argument("--videos", nargs="*", default=list(VIDEOS))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "inference_temporal" / "artifacts")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    num_classes = train_args["num_classes"]
    model = build_model(checkpoint).to(device)

    report = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "device": str(device),
        "num_classes": num_classes,
        "class_names": CLASS_NAMES[:num_classes],
        "best_epoch": checkpoint.get("epoch"),
        "best_validation": checkpoint.get("best"),
        "videos": {},
    }

    with torch.no_grad():
        for video in args.videos:
            dataset = VideoDataset([(video, None)], **dataset_options(train_args))
            thermal, unet_input, target = dataset[0]
            logits = model(thermal[None].to(device), unet_input[None].to(device))
            prediction = logits.argmax(1)[0].cpu().numpy()
            truth = target.numpy()
            video_metrics = metrics(prediction, truth, num_classes)
            report["videos"][video] = video_metrics
            (out / f"{video}_confusion.json").write_text(
                json.dumps({"class_names": CLASS_NAMES[:num_classes], "confusion_matrix": video_metrics["confusion_matrix"]}, indent=2) + "\n"
            )
            render(video, thermal, truth, prediction, out)
            print(f"{video}: macro IoU={video_metrics['macro_iou']:.4f}  Dice={video_metrics['macro_dice']:.4f}  Acc={video_metrics['pixel_accuracy']:.4f}")

    macro_rows = [report["videos"][v] for v in args.videos]
    report["macro_average"] = {
        key: float(np.mean([r[key] for r in macro_rows])) for key in ("macro_iou", "macro_dice", "pixel_accuracy")
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [f"checkpoint: {report['checkpoint']}", f"device: {device}", f"classes: {num_classes}"]
    for v in args.videos:
        r = report["videos"][v]
        lines.append(f"{v}: IoU={r['macro_iou']:.4f} Dice={r['macro_dice']:.4f} Acc={r['pixel_accuracy']:.4f}")
        lines.append("   " + " ".join(f"{k}={v:.3f}" for k, v in r["per_class_iou"].items()))
    lines.append("macro: " + " ".join(f"{k}={v:.4f}" for k, v in report["macro_average"].items()))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
