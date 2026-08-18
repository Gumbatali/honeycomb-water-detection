#!/usr/bin/env python3
"""Evaluate the provided seven-channel U-Net checkpoint on held-out water4."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from PIL import Image
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "honeycomb"
OUT = Path(__file__).resolve().parent / "artifacts"


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetWater(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1, self.e2 = DoubleConv(7, 32), DoubleConv(32, 64)
        self.e3, self.e4 = DoubleConv(64, 128), DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3, self.d3 = nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128)
        self.u2, self.d2 = nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64)
        self.u1, self.d1 = nn.ConvTranspose2d(64, 32, 2, stride=2), DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat((self.u3(e4), e3), dim=1))
        d2 = self.d2(torch.cat((self.u2(d3), e2), dim=1))
        d1 = self.d1(torch.cat((self.u1(d2), e1), dim=1))
        return self.out(d1)


FRAME_INDICES = np.array((10, 30, 50, 70, 100, 150, 200), dtype=int)


def load_preprocessed_stack(video: str) -> tuple[np.ndarray, np.ndarray]:
    """Documented U-Net input: background correction, global scale and panel ROI."""
    all_indices = np.arange(0, int(FRAME_INDICES[-1]) + 1)
    frames = np.stack([np.load(DATA / "images" / "train" / f"{video}_frame_{i:05d}.npy").astype(np.float32)
                       for i in all_indices])
    corrected = np.maximum(frames - frames[0], 0.0)
    corrected /= max(float(corrected.max()), 1e-6)
    diff = corrected[50]
    u8 = np.clip(diff * 255, 0, 255).astype(np.uint8)
    _, otsu = __import__("cv2").threshold(u8, 0, 255, __import__("cv2").THRESH_BINARY + __import__("cv2").THRESH_OTSU)
    roi = ndimage.binary_closing(otsu > 0, iterations=5)
    roi = ndimage.binary_fill_holes(roi)
    roi = ndimage.binary_opening(roi, iterations=3)
    labels, count = ndimage.label(roi)
    if count:
        roi = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    roi = ndimage.binary_erosion(roi, iterations=12)
    return corrected[FRAME_INDICES], roi


def main() -> None:
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetWater().to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "segmentation" / "unet_water_v2.pth", map_location=device,
                                     weights_only=True))
    model.eval()
    video_metrics = {}
    example = None
    with torch.no_grad():
        for video in ("water1", "water2", "water4"):
            inputs, roi = load_preprocessed_stack(video)
            x = torch.from_numpy(inputs).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            target = np.asarray(Image.open(DATA / "masks_binary" / "train" / f"{video}_frame_00050.png")) > 0
            prediction = (prob >= 0.5) & roi
            target &= roi
            tp, fp, fn = int(np.sum(prediction & target)), int(np.sum(prediction & ~target)), int(np.sum(~prediction & target))
            video_metrics[video] = {"dice": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1),
                                    "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "roi_pixels": int(roi.sum())}
            if video == "water4":
                example = (prob, target, prediction, roi)
    held_out = video_metrics["water4"]
    result = {"checkpoint": "models/segmentation/unet_water_v2.pth", "held_out_video": "water4",
              "frames_seconds": (FRAME_INDICES / 10).tolist(),
              "preprocessing": "frame0 subtract; clip; max over 0..20 s; Otsu ROI + close/fill/open/largest/erode12",
              "threshold": 0.5, "dice": held_out["dice"], "iou": held_out["iou"], "precision": held_out["precision"],
              "recall": held_out["recall"], "per_video": video_metrics, "device": str(device)}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "summary.txt").write_text("\n".join(f"{key}: {value}" for key, value in result.items() if key != "frames") + "\n")
    prob, target, prediction, roi = example
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, image, title in zip(axes, (prob, roi, target, prediction), ("U-Net probability", "panel ROI", "binary target in ROI", "prediction @ 0.5")):
        axis.imshow(image, cmap="magma" if title == "U-Net probability" else "gray")
        axis.set_title(title); axis.axis("off")
    fig.suptitle("water4 documented 7-frame preprocessing")
    fig.savefig(OUT / "qualitative_frame.png", dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
