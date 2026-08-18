#!/usr/bin/env python3
"""Tune a full-frame ConvGRU conditioned on frozen U-Net latent features only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "honeycomb"
SIZE, TIME = (240, 320), np.arange(0, 300, 5, dtype=int)
AUGS = ("010_rotation", "011_horizontal_flip", "012_geometric_affine", "013_background_patching",
        "014_defect_location_shift", "015_aggressive_rotation", "016_aggressive_affine")


def load_legacy_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(module)
    return module


SEG = load_legacy_module("seg_eval", ROOT / "experiments" / "019_unet_segmentation_eval" / "evaluate.py")
BASE = load_legacy_module("fullframe_base", ROOT / "experiments" / "020_full_frame_masked_convgru" / "train.py")


def image_dir(video: str, augmentation: str | None) -> tuple[Path, Path]:
    if augmentation is None:
        return DATA / "images" / "train", DATA / "masks" / "train"
    if augmentation.startswith("neutral_patch:"):
        base = DATA / "synthetic" / "neutral_patch" / augmentation.split(":", 1)[1]
        return base / "images", base / "masks"
    base = DATA / "synthetic" / "materialized" / augmentation / video
    return base / "images", base / "masks"


@torch.no_grad()
def latent_features(unet: nn.Module, video: str, augmentation: str | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen U-Net up to e4 and d3; never use its sigmoid mask."""
    directory, _ = image_dir(video, augmentation)
    frames = np.stack([np.load(directory / f"{video}_frame_{i:05d}.npy").astype(np.float32) for i in range(201)])
    corrected = np.maximum(frames - frames[0], 0); corrected /= max(float(corrected.max()), 1e-6)
    x = torch.from_numpy(corrected[SEG.FRAME_INDICES]).unsqueeze(0).to(device)
    e1 = unet.e1(x); e2 = unet.e2(unet.pool(e1)); e3 = unet.e3(unet.pool(e2)); e4 = unet.e4(unet.pool(e3))
    d3 = unet.d3(torch.cat((unet.u3(e4), e3), dim=1))
    return e4.cpu(), d3.cpu()


class LatentVideoDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str | None]], latents: dict[tuple[str, str | None], tuple[torch.Tensor, torch.Tensor]],
                 ignore_label: int | None = None) -> None:
        self.samples, self.latents, self.ignore_label = samples, latents, ignore_label

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, index: int):
        video, augmentation = self.samples[index]; directory, masks = image_dir(video, augmentation)
        frames = np.stack([np.load(directory / f"{video}_frame_{i:05d}.npy").astype(np.float32) for i in TIME])
        thermal = np.maximum(frames - frames[0], 0); thermal /= max(float(thermal.max()), 1e-6)
        thermal = np.stack([cv2.resize(frame, SIZE[::-1], interpolation=cv2.INTER_LINEAR) for frame in thermal])[:, None]
        target = cv2.imread(str(masks / f"{video}_frame_00050.png"), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, SIZE[::-1], interpolation=cv2.INTER_NEAREST)
        if self.ignore_label is not None:
            target[target == self.ignore_label] = 255
        e4, d3 = self.latents[(video, augmentation)]
        return torch.from_numpy(thermal), e4[0], d3[0], torch.from_numpy(target.astype(np.int64))


class LatentConvGRU(nn.Module):
    def __init__(self, hidden: int = 160, layers: int = 2, dropout: float = 0.1, use_d3: bool = True, num_classes: int = 7) -> None:
        super().__init__(); self.hidden, self.layers, self.use_d3 = hidden, layers, use_d3
        self.encoder = nn.Sequential(BASE.Residual(1, 48, 2), BASE.Residual(48, 48), BASE.Residual(48, 96, 2),
                                     BASE.Residual(96, 96), BASE.Residual(96, hidden, 2), BASE.Residual(hidden, hidden))
        self.e4_project = nn.Sequential(nn.Conv2d(256, hidden // 2, 1), nn.GroupNorm(8, hidden // 2), nn.GELU())
        self.d3_project = nn.Sequential(nn.Conv2d(128, hidden // 2, 1), nn.GroupNorm(8, hidden // 2), nn.GELU())
        latent_channels = hidden if use_d3 else hidden // 2
        self.fuse = BASE.Residual(hidden + latent_channels, hidden)
        self.grus = nn.ModuleList([BASE.ConvGRUCell(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout2d(dropout)
        self.decoder = nn.Sequential(BASE.Residual(hidden, 128), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     BASE.Residual(128, 96), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     BASE.Residual(96, 64), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     BASE.Residual(64, 48), nn.Conv2d(48, num_classes, 1))

    def forward(self, thermal: torch.Tensor, e4: torch.Tensor, d3: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = thermal.shape
        encoded = self.encoder(thermal.reshape(batch * steps, 1, height, width)).reshape(batch, steps, self.hidden, height // 8, width // 8)
        target_size = encoded.shape[-2:]
        latent = F.interpolate(self.e4_project(e4), size=target_size, mode="bilinear", align_corners=False)
        if self.use_d3:
            latent = torch.cat((latent, F.interpolate(self.d3_project(d3), size=target_size, mode="bilinear", align_corners=False)), dim=1)
        state = [torch.zeros_like(encoded[:, 0]) for _ in self.grus]
        for step in range(steps):
            current = self.fuse(torch.cat((encoded[:, step], latent), dim=1))
            for layer, gru in enumerate(self.grus):
                state[layer] = gru(current, state[layer]); current = self.dropout(state[layer]) if layer + 1 < self.layers else state[layer]
        return self.decoder(state[-1])


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    classes = logits.shape[1]; valid = target != 255; safe = target.clone(); safe[~valid] = 0
    p = logits.softmax(1)[:, 1:] * valid[:, None]
    y = F.one_hot(safe, classes).permute(0, 3, 1, 2).float()[:, 1:] * valid[:, None]
    return 1 - ((2 * (p * y).sum((0, 2, 3)) + 1) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + 1)).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval(); inter = np.zeros(num_classes - 1); union = np.zeros(num_classes - 1)
    for thermal, e4, d3, target in loader:
        pred = model(thermal.to(device), e4.to(device), d3.to(device)).argmax(1).cpu().numpy(); truth = target.numpy()
        valid = truth != 255
        for cls in range(1, num_classes):
            inter[cls - 1] += np.sum((pred == cls) & (truth == cls) & valid)
            union[cls - 1] += np.sum(((pred == cls) | (truth == cls)) & valid)
    return {"macro_iou": float((inter / np.maximum(union, 1)).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--layers", type=int, choices=(1, 2), default=2); parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--use-d3", action="store_true"); parser.add_argument("--train-video", default="water1"); parser.add_argument("--valid-video", default="water2")
    parser.add_argument("--num-classes", type=int, default=7); parser.add_argument("--ignore-label", type=int)
    parser.add_argument("--include-neutral-patch", action="store_true")
    parser.add_argument("--include-patched-only", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unet-checkpoint", type=Path, default=ROOT / "models" / "segmentation" / "v1" / "unet_water_v2.pth")
    args = parser.parse_args()
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda"); unet = SEG.UNetWater().to(device).eval()
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device, weights_only=True))
    train_samples = [(args.train_video, aug) for aug in (None, *AUGS)]
    if args.include_neutral_patch or args.include_patched_only:
        if args.train_video != "water1": raise ValueError("neutral/patch videos currently derive from water1")
        names = ("patched_shift_left_down", "patched_shift_right_up", "patched_shift_left_up")
        if args.include_neutral_patch:
            names = ("neutral_water1", *names)
        train_samples.extend(("water1", f"neutral_patch:{name}") for name in names)
    valid_samples = [(args.valid_video, None)]
    latents = {sample: latent_features(unet, *sample, device) for sample in (*train_samples, *valid_samples)}
    train_loader = DataLoader(LatentVideoDataset(train_samples, latents, args.ignore_label), batch_size=1, shuffle=True)
    valid_loader = DataLoader(LatentVideoDataset(valid_samples, latents, args.ignore_label), batch_size=1)
    model = LatentConvGRU(args.hidden, args.layers, args.dropout, args.use_d3, args.num_classes).to(device)
    loaded_keys = 0
    if args.init_checkpoint:
        previous = torch.load(args.init_checkpoint, map_location=device, weights_only=False)["model_state"]
        current = model.state_dict(); compatible = {key: value for key, value in previous.items() if key in current and current[key].shape == value.shape}
        loaded_keys = len(compatible); model.load_state_dict(compatible, strict=False)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=.5, patience=8, min_lr=3e-6)
    weights = torch.tensor((.08,) + (1.,) * (args.num_classes - 1), device=device)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True); best = {"macro_iou": -1.}; history = []; stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for thermal, e4, d3, target in train_loader:
            logits = model(thermal.to(device), e4.to(device), d3.to(device)); target = target.to(device)
            loss = F.cross_entropy(logits, target, weight=weights, ignore_index=255) + .5 * dice_loss(logits, target)
            optim.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optim.step()
        score = evaluate(model, valid_loader, device, args.num_classes); score["epoch"] = epoch; score["lr"] = optim.param_groups[0]["lr"]; history.append(score); scheduler.step(score["macro_iou"])
        if score["macro_iou"] > best["macro_iou"] + .002:
            best = score.copy(); stale = 0; torch.save({"model_state": model.state_dict(), "args": vars(args), "best": best}, out / "best.pt")
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0: print(f"epoch={epoch:03d} macro_iou={score['macro_iou']:.4f} lr={score['lr']:.1e}")
        if stale >= 15: print(f"early_stop={epoch}"); break
    report = {"architecture": "frozen U-Net latent fusion + ConvGRU", "args": vars(args), "best_validation": best,
              "history": history, "epochs_ran": len(history), "train_samples": len(train_samples), "warm_start_keys": loaded_keys}
    report["args"]["output_dir"] = str(args.output_dir); report["args"]["unet_checkpoint"] = str(args.unet_checkpoint)
    report["args"]["init_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint else None
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n"); (out / "summary.txt").write_text(json.dumps(best, indent=2) + "\n")


if __name__ == "__main__": main()
