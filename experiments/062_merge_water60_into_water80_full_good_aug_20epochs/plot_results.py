#!/usr/bin/env python3
"""Plot learning history and per-class validation IoU for experiment 062."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CLASS_NAMES = ("water20", "water40", "water60–80", "water100")


def main() -> None:
    report = json.loads((ARTIFACTS / "metrics.json").read_text())
    history = report["history"]
    epochs = np.array([row["epoch"] for row in history])
    best_epoch = report["best_validation"]["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    axes[0].plot(epochs[1:], [row["train_loss"] for row in history[1:]], marker="o", label="train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="water2 validation loss")
    axes[0].axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axes[0].grid(alpha=.25); axes[0].legend()

    axes[1].plot(epochs, [row["macro_iou"] for row in history], marker="o", label="macro IoU")
    axes[1].plot(epochs, [row["macro_dice"] for row in history], marker="o", label="macro Dice")
    axes[1].axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Water2 validation quality", ylim=(0, 1))
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.savefig(ARTIFACTS / "learning_curves.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for index, name in enumerate(CLASS_NAMES, 1):
        axis.plot(epochs, [row[f"iou_class_{index}"] for row in history], marker="o", label=name)
    axis.axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
    axis.set(xlabel="Epoch", ylabel="IoU", title="Water2 validation IoU by deployed class", ylim=(0, 1))
    axis.grid(alpha=.25); axis.legend(ncol=2)
    fig.savefig(ARTIFACTS / "per_class_iou.png", dpi=180); plt.close(fig)

    best = report["best_validation"]
    lines = [f"best epoch: {best_epoch}", "", "class          validation IoU"]
    for index, name in enumerate(CLASS_NAMES, 1):
        lines.append(f"{name:14s} {best[f'iou_class_{index}']:.6f}")
    lines.append(f"{'macro':14s} {best['macro_iou']:.6f}")
    (ARTIFACTS / "per_class_iou_best_epoch.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
