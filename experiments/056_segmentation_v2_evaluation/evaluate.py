#!/usr/bin/env python3
"""Evaluate segmentation-v2 on all real labelled videos."""
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "019_unet_segmentation_eval" / "evaluate.py"
spec = importlib.util.spec_from_file_location("segmentation_v1_evaluator", SOURCE)
if spec is None or spec.loader is None: raise ImportError(f"cannot load {SOURCE}")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
load_preprocessed_stack = module.load_preprocessed_stack

DATA, OUT, VIDEOS = ROOT / "data" / "honeycomb", Path(__file__).resolve().parent / "artifacts", ("water1", "water2", "water4")


class V2DoubleConv(nn.Module):
    """v2 inserts a parameter-free layer between ReLU and its second convolution."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                  nn.Dropout2d(.1), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)


class V2UNetWater(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1, self.e2 = V2DoubleConv(7, 32), V2DoubleConv(32, 64)
        self.e3, self.e4 = V2DoubleConv(64, 128), V2DoubleConv(128, 256); self.pool = nn.MaxPool2d(2)
        self.u3, self.d3 = nn.ConvTranspose2d(256, 128, 2, stride=2), V2DoubleConv(256, 128)
        self.u2, self.d2 = nn.ConvTranspose2d(128, 64, 2, stride=2), V2DoubleConv(128, 64)
        self.u1, self.d1 = nn.ConvTranspose2d(64, 32, 2, stride=2), V2DoubleConv(64, 32); self.out = nn.Conv2d(32, 1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat((self.u3(e4), e3), 1)); d2 = self.d2(torch.cat((self.u2(d3), e2), 1))
        return self.out(self.d1(torch.cat((self.u1(d2), e1), 1)))


def score(prob: np.ndarray, target: np.ndarray, roi: np.ndarray, threshold: float) -> dict[str, float]:
    pred, truth = (prob >= threshold) & roi, target & roi
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth & roi).sum())
    return {"threshold": threshold, "dice": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "pixel_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "roi_pixels": int(roi.sum()), "defect_pixels": int(truth.sum()), "predicted_defect_pixels": int(pred.sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ROOT / "models" / "segmentation" / "v2" / "unet_water_v2.pth"
    model = V2UNetWater().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    report: dict[str, object] = {"checkpoint": str(checkpoint.relative_to(ROOT)), "device": str(device),
        "model": "binary 7-channel U-Net", "primary_threshold": 0.5, "videos": {}}
    examples, thresholds = [], np.linspace(.05, .95, 19)
    with torch.no_grad():
        for video in VIDEOS:
            inputs, roi = load_preprocessed_stack(video)
            prob = torch.sigmoid(model(torch.from_numpy(inputs)[None].to(device)))[0, 0].cpu().numpy()
            target = np.asarray(Image.open(DATA / "masks_binary" / "train" / f"{video}_frame_00050.png")) > 0
            fixed, sweep = score(prob, target, roi, .5), [score(prob, target, roi, float(t)) for t in thresholds]
            report["videos"][video] = {"threshold_0_5": fixed, "diagnostic_best_dice_threshold": max(sweep, key=lambda x: x["dice"]), "threshold_sweep": sweep}
            examples.append((video, inputs[2], prob, target & roi, (prob >= .5) & roi, roi))
    fixed_rows = [report["videos"][v]["threshold_0_5"] for v in VIDEOS]
    report["macro_fixed_threshold"] = {key: float(np.mean([r[key] for r in fixed_rows])) for key in ("dice", "iou", "precision", "recall", "specificity", "pixel_accuracy")}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"checkpoint: {report['checkpoint']}", f"device: {device}", "fixed threshold: 0.50"]
    for video in VIDEOS:
        r = report["videos"][video]["threshold_0_5"]
        lines.append(f"{video}: Dice={r['dice']:.6f} IoU={r['iou']:.6f} P={r['precision']:.6f} R={r['recall']:.6f} Acc={r['pixel_accuracy']:.6f}")
    lines.append("macro: " + " ".join(f"{k}={v:.6f}" for k, v in report["macro_fixed_threshold"].items()))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(len(examples), 5, figsize=(18, 3.6 * len(examples)), constrained_layout=True)
    for row, (video, thermal, prob, target, pred, roi) in zip(np.atleast_2d(axes), examples):
        for axis, (image, title, cmap) in zip(row, ((thermal, "thermal at 5 s", "inferno"), (prob, "v2 probability", "magma"), (roi, "panel ROI", "gray"), (target, "binary target", "gray"), (pred, "prediction @ 0.50", "gray"))):
            axis.imshow(image, cmap=cmap); axis.set_title(f"{video}: {title}"); axis.axis("off")
    fig.savefig(OUT / "all_videos_validation.png", dpi=170); plt.close(fig)
    print((OUT / "summary.txt").read_text(), end="")


if __name__ == "__main__": main()
