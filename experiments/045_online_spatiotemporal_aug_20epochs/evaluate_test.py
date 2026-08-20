#!/usr/bin/env python3
"""One-shot evaluation of experiment 045 on untouched water4."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "030_unet_feature_pyramid_convgru"
sys.path.insert(0, str(SOURCE))

from model import UNetFeatureConvGRU, UNetFeatureConvGRUMultitask  # noqa: E402
from train import VideoDataset, evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    cli = parser.parse_args()
    artifact = cli.artifact_dir.resolve() if cli.artifact_dir else Path(__file__).resolve().parent / "artifacts"
    checkpoint = torch.load(artifact / "best.pt", map_location="cuda", weights_only=False)
    args = checkpoint["args"]
    device = torch.device("cuda")
    model_class = UNetFeatureConvGRUMultitask if args.get("multitask_heads", False) else UNetFeatureConvGRU
    thermal_channels = 2 if args.get("thermal_representation", "absolute") == "both" else 1
    model = model_class(args["hidden"], args["dropout"], args["num_classes"], thermal_channels,
                        args.get("separate_thermal_stems", False)).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])

    dataset = VideoDataset(
        [("water4", None)],
        args["ignore_label"],
        args["thermal_normalization"],
        args["apply_roi"],
        merge_label=args.get("merge_label"),
        merge_into=args.get("merge_into"),
        compact_after_merge=args.get("compact_after_merge", False),
        neutralize_label=args.get("neutralize_label"),
        thermal_representation=args.get("thermal_representation", "absolute"),
        contrast_normalization=args.get("contrast_normalization", "pixel_peak"),
    )
    class_weights = [0.08] + [1.0] * (args["num_classes"] - 1)
    if args.get("merge_into") is not None:
        merged_index = args["merge_into"] - int(args.get("compact_after_merge", False))
        class_weights[merged_index] = args.get("merged_class_weight", 1.0)
    weights = torch.tensor(class_weights, device=device)
    auxiliary = ((args.get("binary_loss_weight", 0.30), args.get("ordinal_loss_weight", 0.20))
                 if args.get("multitask_heads", False) else (0.0, 0.0))
    metrics = evaluate(model, torch.utils.data.DataLoader(dataset, batch_size=1), device, weights,
                       args["num_classes"], *auxiliary)
    metrics = {("test_" + key[4:] if key.startswith("val_") else key): value for key, value in metrics.items()}
    metrics.update(test_video="water4", checkpoint_epoch=checkpoint["best"]["epoch"])
    (artifact / "water4_test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    thermal, unet_input, target = dataset[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(thermal[None].to(device), unet_input[None].to(device))
        logits = output[0] if isinstance(output, tuple) else output
        prediction = logits.argmax(1)[0].cpu().numpy()
    truth = target.numpy()
    valid = truth != 255
    prediction[~valid] = 0
    confusion = np.zeros((args["num_classes"], args["num_classes"]), dtype=np.int64)
    for true_class in range(args["num_classes"]):
        for predicted_class in range(args["num_classes"]):
            confusion[true_class, predicted_class] = np.sum(
                (truth == true_class) & (prediction == predicted_class) & valid
            )
    (artifact / "water4_test_confusion_matrix.json").write_text(json.dumps(confusion.tolist(), indent=2) + "\n")

    display_truth = truth.copy()
    display_truth[~valid] = 0
    error = (prediction != truth) & valid
    confidence = logits.softmax(1).amax(1)[0].float().cpu().numpy()
    confidence[~valid] = 0
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
    panels = (
        (thermal[10, 0], "thermal at 5 s", "inferno", None),
        (display_truth, "water4 target", "tab10", 5),
        (prediction, "prediction", "tab10", 5),
        (error, "pixel error", "Reds", 1),
        (confidence, "max probability", "viridis", 1),
    )
    for axis, (image, title, cmap, vmax) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0 if vmax is not None else None, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    protocol = "water4 test with water120 neutralized" if args.get("neutralize_label") is not None else "Untouched water4 test"
    fig.suptitle(f"{protocol}, checkpoint epoch {checkpoint['best']['epoch']}, macro-IoU {metrics['macro_iou']:.4f}")
    fig.savefig(artifact / "water4_test.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
