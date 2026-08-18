#!/usr/bin/env python3
"""Render the best experiment-039 prediction on untouched water2 validation."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "030_unet_feature_pyramid_convgru"
sys.path.insert(0, str(SOURCE))
from model import UNetFeatureConvGRU  # noqa: E402
from train import VideoDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact-dir", type=Path)
    cli = parser.parse_args()
    artifact = cli.artifact_dir.resolve() if cli.artifact_dir else Path(__file__).resolve().parent / "artifacts"
    checkpoint = torch.load(artifact / "best.pt", map_location="cuda", weights_only=False)
    args = checkpoint["args"]
    model = UNetFeatureConvGRU(args["hidden"], args["dropout"], args["num_classes"]).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])
    thermal, unet_input, target = VideoDataset([("water2", None)], args["ignore_label"],
        args.get("thermal_normalization", "global"), args.get("apply_roi", False),
        merge_label=args.get("merge_label"), merge_into=args.get("merge_into"))[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(thermal[None].cuda(), unet_input[None].cuda()).argmax(1)[0].cpu().numpy()
    truth = target.numpy(); valid = truth != 255
    prediction[~valid] = 0  # deployment post-processing: no defects outside the panel ROI
    error = (prediction != truth) & valid
    confusion = np.zeros((args["num_classes"], args["num_classes"]), dtype=np.int64)
    for true_class in range(args["num_classes"]):
        for predicted_class in range(args["num_classes"]):
            confusion[true_class, predicted_class] = np.sum((truth == true_class) & (prediction == predicted_class) & valid)
    (artifact / "water2_confusion_matrix.json").write_text(json.dumps(confusion.tolist(), indent=2) + "\n")
    display_truth = truth.copy(); display_truth[~valid] = 0
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    panels = ((thermal[10, 0], "thermal at 5 s", "inferno", None),
              (display_truth, "water2 target", "tab10", 5),
              (prediction, "prediction", "tab10", 5), (error, "pixel error", "Reds", 1))
    for axis, (image, title, cmap, vmax) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0 if vmax is not None else None, vmax=vmax)
        axis.set_title(title); axis.axis("off")
    fig.suptitle(f"{artifact.parent.name}, best epoch {checkpoint['best']['epoch']}, macro-IoU {checkpoint['best']['macro_iou']:.4f}")
    fig.savefig(artifact / "water2_validation.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
