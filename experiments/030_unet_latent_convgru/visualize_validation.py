#!/usr/bin/env python3
"""Render the held-out water2 prediction for a latent-ConvGRU checkpoint."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("latent_training", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "experiments" / "033_latent_gru_1x128" / "artifacts" / "best.pt")
    parser.add_argument("--video", default="water2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ignore-label", type=int)
    cli = parser.parse_args()
    module = load_module(HERE / "train.py")
    checkpoint_path = cli.checkpoint.resolve()
    output = cli.output.resolve() if cli.output else checkpoint_path.parent / f"{cli.video}_validation.png"
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    unet = module.SEG.UNetWater().to(device).eval()
    unet.load_state_dict(torch.load(ROOT / "models" / "segmentation" / "v1" / "unet_water_v2.pth", map_location=device, weights_only=True))
    latent = module.latent_features(unet, cli.video, None, device)
    ignore_label = cli.ignore_label if cli.ignore_label is not None else args.get("ignore_label")
    dataset = module.LatentVideoDataset([(cli.video, None)], {(cli.video, None): latent}, ignore_label)
    thermal, e4, d3, target = dataset[0]
    num_classes = args.get("num_classes", 7)
    model = module.LatentConvGRU(args["hidden"], args["layers"], args["dropout"], args["use_d3"], num_classes).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad(): prediction = model(thermal.unsqueeze(0).to(device), e4.unsqueeze(0).to(device), d3.unsqueeze(0).to(device))[0].argmax(0).cpu().numpy()
    truth = target.numpy(); thermal_5s = thermal[10, 0].numpy(); valid = truth != 255
    error = ((prediction != truth) & valid).astype(np.uint8)
    per_class = []
    for cls in range(1, num_classes):
        intersection = int(np.sum((prediction == cls) & (truth == cls) & valid))
        union = int(np.sum(((prediction == cls) | (truth == cls)) & valid))
        per_class.append(intersection / max(union, 1))
    metrics = {"macro_iou": float(np.mean(per_class)), "per_class_iou": per_class, "classes": list(range(1, num_classes))}
    (output.parent / f"{cli.video}_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    panels = ((thermal_5s, "thermal at 5 s", "inferno", None, None), (truth, f"{cli.video} label", "tab10", 0, num_classes - 1),
              (prediction, "latent ConvGRU prediction", "tab10", 0, num_classes - 1), (error, "pixel error", "Reds", 0, 1))
    for axis, (image, title, cmap, low, high) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=low, vmax=high); axis.set_title(title); axis.axis("off")
    best = checkpoint.get("best", {})
    fig.suptitle(f"Validation: U-Net v1 e4 + {args['layers']}x{args['hidden']} ConvGRU, epoch {best.get('epoch', '?')}")
    fig.savefig(output, dpi=160); plt.close(fig)
    print(output)


if __name__ == "__main__": main()
