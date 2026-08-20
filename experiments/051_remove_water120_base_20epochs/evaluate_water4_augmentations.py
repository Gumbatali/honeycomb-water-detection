#!/usr/bin/env python3
"""Robustness evaluation of experiment 051 on materialized water4 variants.

Each variant contains jointly transformed thermal frames and semantic targets.
VideoDataset is deliberately reused so the test follows experiment 051's exact
segmentation pre-processing: water120 response neutralization, pixel-peak
normalization, derived ROI and seven-frame U-Net input preparation.
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "030_unet_feature_pyramid_convgru"
sys.path.insert(0, str(SOURCE))

from model import UNetFeatureConvGRU, UNetFeatureConvGRUMultitask  # noqa: E402
from train import VideoDataset, evaluate  # noqa: E402


VARIANTS = (
    ("baseline", None, "Original water4"),
    ("010_rotation", "010_rotation", "Small rotation (+6°)"),
    ("011_horizontal_flip", "011_horizontal_flip", "Horizontal flip"),
    ("012_geometric_affine", "012_geometric_affine", "Mild affine"),
    ("013_background_patching", "013_background_patching", "Background patches"),
    ("014_defect_location_shift", "014_defect_location_shift", "Defect-location shift"),
    ("015_aggressive_rotation", "015_aggressive_rotation", "Aggressive rotation (+15°)"),
    ("016_aggressive_affine", "016_aggressive_affine", "Aggressive affine"),
)


def load_model(artifact: Path) -> tuple[torch.nn.Module, dict, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(artifact / "best.pt", map_location=device, weights_only=False)
    args = checkpoint["args"]
    cls = UNetFeatureConvGRUMultitask if args.get("multitask_heads", False) else UNetFeatureConvGRU
    channels = 2 if args.get("thermal_representation", "absolute") == "both" else 1
    model = cls(args["hidden"], args["dropout"], args["num_classes"], channels,
                args.get("separate_thermal_stems", False)).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint, device


def dataset_for(video: str, augmentation: str | None, args: dict) -> VideoDataset:
    return VideoDataset(
        [(video, augmentation)], args["ignore_label"], args["thermal_normalization"], args["apply_roi"],
        merge_label=args.get("merge_label"), merge_into=args.get("merge_into"),
        compact_after_merge=args.get("compact_after_merge", False),
        neutralize_label=args.get("neutralize_label"),
        thermal_representation=args.get("thermal_representation", "absolute"),
        contrast_normalization=args.get("contrast_normalization", "pixel_peak"),
    )


def overview(output: Path, title: str, thermal: torch.Tensor, truth: np.ndarray,
             prediction: np.ndarray, confidence: np.ndarray) -> None:
    valid = truth != 255
    shown_truth = truth.copy(); shown_truth[~valid] = 0
    shown_prediction = prediction.copy(); shown_prediction[~valid] = 0
    error = (prediction != truth) & valid
    panels = (
        (thermal[10, 0].numpy(), "Thermal, 5 s", "inferno", None),
        (shown_truth, "Target", "tab10", 5),
        (shown_prediction, "Prediction", "tab10", 5),
        (error, "Pixel error", "Reds", 1),
        (confidence, "Max probability", "viridis", 1),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(22, 4), constrained_layout=True)
    for axis, (image, label, cmap, vmax) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0 if vmax is not None else None, vmax=vmax)
        axis.set_title(label); axis.axis("off")
    fig.suptitle(title)
    fig.savefig(output, dpi=170); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", choices=("water2", "water4"), default="water4",
                        help="labelled source video to evaluate")
    cli = parser.parse_args()
    video = cli.video
    artifact = Path(__file__).resolve().parent / "artifacts"
    output = artifact / f"{video}_augmentation_robustness"; output.mkdir(parents=True, exist_ok=True)
    model, checkpoint, device = load_model(artifact)
    args = checkpoint["args"]
    weights = torch.tensor([0.08] + [1.0] * (args["num_classes"] - 1), device=device)
    if args.get("merge_into") is not None:
        merged_index = args["merge_into"] - int(args.get("compact_after_merge", False))
        weights[merged_index] = args.get("merged_class_weight", 1.0)
    auxiliary = ((args.get("binary_loss_weight", .30), args.get("ordinal_loss_weight", .20))
                 if args.get("multitask_heads", False) else (0.0, 0.0))
    report: dict[str, object] = {"video": video, "checkpoint": "artifacts/best.pt", "checkpoint_epoch": checkpoint["best"]["epoch"],
                                 "device": str(device), "preprocessing": {key: args.get(key) for key in
                                 ("ignore_label", "neutralize_label", "thermal_normalization", "apply_roi",
                                  "thermal_representation", "contrast_normalization")}, "variants": {}}
    suite_examples = []
    for identifier, augmentation, label in VARIANTS:
        dataset = dataset_for(video, augmentation, args)
        metrics = evaluate(model, DataLoader(dataset, batch_size=1), device, weights, args["num_classes"], *auxiliary)
        thermal, unet_input, target = dataset[0]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model(thermal[None].to(device), unet_input[None].to(device))
            logits = result[0] if isinstance(result, tuple) else result
            probability = logits.softmax(1)[0].float().cpu().numpy()
        prediction, confidence, truth = probability.argmax(0), probability.max(0), target.numpy()
        variant_dir = output / identifier; variant_dir.mkdir(exist_ok=True)
        overview(variant_dir / "overview.png", f"{video} — {label}; macro-IoU {metrics['macro_iou']:.4f}",
                 thermal, truth, prediction, confidence)
        confusion = np.zeros((args["num_classes"], args["num_classes"]), dtype=np.int64)
        valid = truth != 255
        for actual in range(args["num_classes"]):
            for predicted in range(args["num_classes"]):
                confusion[actual, predicted] = np.sum((truth == actual) & (prediction == predicted) & valid)
        (variant_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (variant_dir / "confusion_matrix.json").write_text(json.dumps(confusion.tolist(), indent=2) + "\n")
        report["variants"][identifier] = {"label": label, "augmentation": augmentation, **metrics}
        suite_examples.append((label, thermal[10, 0].numpy(), truth, prediction, metrics["macro_iou"]))
        print(f"{identifier:28s} macro_iou={metrics['macro_iou']:.6f} macro_dice={metrics['macro_dice']:.6f}", flush=True)
    original_iou = report["variants"]["baseline"]["macro_iou"]
    for key, value in report["variants"].items(): value["macro_iou_delta_from_baseline"] = value["macro_iou"] - original_iou
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"video: {video}", f"checkpoint epoch: {checkpoint['best']['epoch']}", "pre-processing: water120 neutralization + pixel-peak + ROI + U-Net preparation", "",
             f"{'variant':28s} {'macro IoU':>10s} {'macro Dice':>11s} {'delta':>10s}"]
    for key, value in report["variants"].items():
        lines.append(f"{key:28s} {value['macro_iou']:10.6f} {value['macro_dice']:11.6f} {value['macro_iou_delta_from_baseline']:+10.6f}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(len(suite_examples), 3, figsize=(13, 3.2 * len(suite_examples)), constrained_layout=True)
    for row, (label, thermal, truth, prediction, score) in zip(np.atleast_2d(axes), suite_examples):
        valid = truth != 255; truth = truth.copy(); truth[~valid] = 0; prediction = prediction.copy(); prediction[~valid] = 0
        for axis, image, panel, cmap, vmax in zip(row, (thermal, truth, prediction), ("Thermal 5 s", "Target", "Prediction"),
                                                   ("inferno", "tab10", "tab10"), (None, 5, 5)):
            axis.imshow(image, cmap=cmap, vmin=0 if vmax is not None else None, vmax=vmax); axis.set_title(panel); axis.axis("off")
        row[0].set_ylabel(f"{label}\nIoU={score:.4f}", rotation=0, ha="right", va="center")
    fig.savefig(output / "suite_overview.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
