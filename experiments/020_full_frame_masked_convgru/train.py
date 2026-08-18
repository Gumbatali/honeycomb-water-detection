#!/usr/bin/env python3
"""Full-frame 30-second mask-conditioned ConvGRU semantic segmenter."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "honeycomb"
MANIFESTS = ROOT / "data" / "synthetic" / "video_augmentation_manifests"
SIZE = (240, 320)  # H, W
TIME = np.arange(0, 300, 5, dtype=int)  # 60 frames, 0.5-second interval
ALL_AUGS = ("010_rotation", "011_horizontal_flip", "012_geometric_affine", "013_background_patching",
            "014_defect_location_shift", "015_aggressive_rotation", "016_aggressive_affine")


def segmentation_module():
    spec = importlib.util.spec_from_file_location("seg_eval", ROOT / "experiments" / "019_unet_segmentation_eval" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Residual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(),
                                  nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((x, state), dim=1)).chunk(2, dim=1)
        reset, update = torch.sigmoid(reset), torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * state), dim=1)))
        return (1 - update) * state + update * candidate


class FullFrameConvGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(Residual(2, 48, 2), Residual(48, 48), Residual(48, 96, 2), Residual(96, 96),
                                     Residual(96, 160, 2), Residual(160, 160))
        self.temporal = ConvGRUCell(160)
        self.decoder = nn.Sequential(Residual(160, 128), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     Residual(128, 96), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     Residual(96, 64), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     Residual(64, 48), nn.Conv2d(48, 7, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        encoded = self.encoder(x.reshape(batch * steps, channels, height, width)).reshape(batch, steps, 160, height // 8, width // 8)
        state = torch.zeros_like(encoded[:, 0])
        for index in range(steps):
            state = self.temporal(encoded[:, index], state)
        return self.decoder(state)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)[:, 1:]
    one_hot = F.one_hot(target, 7).permute(0, 3, 1, 2).float()[:, 1:]
    numerator = 2 * (probabilities * one_hot).sum(dim=(0, 2, 3)) + 1
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3)) + 1
    return 1 - (numerator / denominator).mean()


def transform_probability(probability: np.ndarray, augmentation: str | None) -> np.ndarray:
    if augmentation is None:
        return probability
    info = json.loads((MANIFESTS / augmentation / "water1.json").read_text())["augmentation"]
    matrix = info["matrix_2x3"]
    if matrix is None:
        return probability
    return cv2.warpAffine(probability, np.asarray(matrix, np.float32), (probability.shape[1], probability.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class VideoDataset(Dataset):
    def __init__(self, videos: list[tuple[str, str | None]], probabilities: dict[tuple[str, str | None], np.ndarray]) -> None:
        self.videos, self.probabilities = videos, probabilities

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video, augmentation = self.videos[index]
        if augmentation is None:
            image_dir, mask_dir = DATA / "images" / "train", DATA / "masks" / "train"
        else:
            base = DATA / "synthetic" / "materialized" / augmentation / video
            image_dir, mask_dir = base / "images", base / "masks"
        frames = np.stack([np.load(image_dir / f"{video}_frame_{time:05d}.npy").astype(np.float32) for time in TIME])
        thermal = np.maximum(frames - frames[0], 0)
        thermal /= max(float(thermal.max()), 1e-6)
        thermal = np.stack([cv2.resize(frame, SIZE[::-1], interpolation=cv2.INTER_LINEAR) for frame in thermal])
        probability = cv2.resize(self.probabilities[(video, augmentation)], SIZE[::-1], interpolation=cv2.INTER_LINEAR)
        x = np.stack((thermal, np.broadcast_to(probability, thermal.shape)), axis=1).astype(np.float32)
        target = cv2.resize(cv2.imread(str(mask_dir / f"{video}_frame_00050.png"), cv2.IMREAD_GRAYSCALE), SIZE[::-1], interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(x), torch.from_numpy(target.astype(np.int64))


@torch.no_grad()
def metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); intersections = np.zeros(6); unions = np.zeros(6); correct = pixels = 0
    for x, target in loader:
        prediction = model(x.to(device)).argmax(dim=1).cpu().numpy(); truth = target.numpy()
        correct += int((prediction == truth).sum()); pixels += prediction.size
        for cls in range(1, 7):
            intersections[cls - 1] += np.sum((prediction == cls) & (truth == cls))
            unions[cls - 1] += np.sum((prediction == cls) | (truth == cls))
    iou = intersections / np.maximum(unions, 1)
    return {"macro_iou": float(iou.mean()), "pixel_accuracy": correct / pixels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--train-augmentation", action="append", default=[],
                        help="materialized augmentation ID; repeat to add several")
    parser.add_argument("--init-checkpoint", type=Path, help="best.pt used to initialise a curriculum stage")
    args = parser.parse_args(); random.seed(17); np.random.seed(17); torch.manual_seed(17)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda"); seg = segmentation_module(); unet = seg.UNetWater().to(device).eval()
    unet.load_state_dict(torch.load(ROOT / "models" / "segmentation" / "unet_water_v2.pth", map_location=device, weights_only=True))
    probabilities = {}
    with torch.no_grad():
        for video in ("water1", "water2", "water4"):
            source, _ = seg.load_preprocessed_stack(video)
            probability = torch.sigmoid(unet(torch.from_numpy(source).unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            for augmentation in (None, *ALL_AUGS): probabilities[(video, augmentation)] = transform_probability(probability, augmentation)
    selected_augs = tuple(args.train_augmentation) if args.train_augmentation else ALL_AUGS
    unknown = set(selected_augs) - set(ALL_AUGS)
    if unknown: raise ValueError(f"Unknown augmentation IDs: {sorted(unknown)}")
    train_videos = [(video, augmentation) for video in ("water1", "water2") for augmentation in (None, *selected_augs)]
    train_loader = DataLoader(VideoDataset(train_videos, probabilities), batch_size=1, shuffle=True)
    valid_loader = DataLoader(VideoDataset([("water4", None)], probabilities), batch_size=1)
    model = FullFrameConvGRU().to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=False)["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = torch.tensor((0.08, 1, 1, 1, 1, 1, 1), device=device); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    history = []; best = {"macro_iou": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, target in train_loader:
            logits = model(x.to(device)); target = target.to(device)
            loss = F.cross_entropy(logits, target, weight=weights) + 0.5 * soft_dice_loss(logits, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        current = metrics(model, valid_loader, device); current["epoch"] = epoch; history.append(current)
        if current["macro_iou"] > best["macro_iou"]:
            best = current.copy(); torch.save({"model_state": model.state_dict(), "best": best}, output / "best.pt")
        if epoch == 1 or epoch % 10 == 0: print(f"epoch={epoch:03d} macro_iou={current['macro_iou']:.4f} acc={current['pixel_accuracy']:.4f}")
    report = {"architecture": "full-frame mask-conditioned ConvGRU", "temporal_frames": TIME.tolist(), "seconds": 30,
              "train_sequences": len(train_videos), "train_augmentations": selected_augs,
              "initialisation": str(args.init_checkpoint) if args.init_checkpoint else None,
              "best_validation": best, "history": history}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.txt").write_text(json.dumps(best, indent=2) + "\n")
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=False)["model_state"])
    x, target = next(iter(valid_loader))
    with torch.no_grad(): prediction = model(x.to(device)).argmax(1)[0].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, image, title in zip(axes, (target[0].numpy(), prediction), ("water4 target", "full-frame ConvGRU prediction")):
        axis.imshow(image, cmap="tab10", vmin=0, vmax=6); axis.set_title(title); axis.axis("off")
    fig.savefig(output / "water4_prediction.png", dpi=150); plt.close(fig)


if __name__ == "__main__": main()
