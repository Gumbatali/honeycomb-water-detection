#!/usr/bin/env python3
"""One-shot evaluation of experiment 045 on untouched water4."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "030_unet_feature_pyramid_convgru"
sys.path.insert(0, str(SOURCE))

from model import UNetFeatureConvGRU  # noqa: E402
from train import VideoDataset, evaluate  # noqa: E402


def main() -> None:
    artifact = Path(__file__).resolve().parent / "artifacts"
    checkpoint = torch.load(artifact / "best.pt", map_location="cuda", weights_only=False)
    args = checkpoint["args"]
    device = torch.device("cuda")
    model = UNetFeatureConvGRU(args["hidden"], args["dropout"], args["num_classes"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])

    dataset = VideoDataset(
        [("water4", None)],
        args["ignore_label"],
        args["thermal_normalization"],
        args["apply_roi"],
    )
    weights = torch.tensor((0.08,) + (1,) * (args["num_classes"] - 1), device=device)
    metrics = evaluate(model, torch.utils.data.DataLoader(dataset, batch_size=1), device, weights, args["num_classes"])
    metrics = {("test_" + key[4:] if key.startswith("val_") else key): value for key, value in metrics.items()}
    metrics.update(test_video="water4", checkpoint_epoch=checkpoint["best"]["epoch"])
    (artifact / "water4_test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    thermal, unet_input, target = dataset[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(thermal[None].to(device), unet_input[None].to(device))
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
    fig.suptitle(
        f"Untouched water4 test, checkpoint epoch {checkpoint['best']['epoch']}, "
        f"macro-IoU {metrics['macro_iou']:.4f}"
    )
    fig.savefig(artifact / "water4_test.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
