#!/usr/bin/env python3
"""Plot the controlled contrast experiments against the water120-free baseline."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
CURRENT = Path(__file__).resolve().parent / "artifacts"


def best(name: str) -> float:
    report = json.loads((EXP / name / "artifacts" / "metrics.json").read_text())
    return float(report["best_validation"]["macro_iou"])


def main() -> None:
    report = json.loads((CURRENT / "metrics.json").read_text())
    history = report["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(epochs, [row["macro_iou"] for row in history], marker="o", label="water2 macro-IoU")
    axes[0].axhline(best("051_remove_water120_base_20epochs"), color="black", ls="--", label="051 baseline")
    axes[0].set(xlabel="epoch", ylabel="macro-IoU", title="055 validation trajectory", ylim=(0, 0.8))
    axes[0].legend()
    names = ("051 baseline", "053 contrast", "054 both", "055 ROI-scale + stems")
    values = (best("051_remove_water120_base_20epochs"), best("053_local_contrast_only_20epochs"),
              best("054_absolute_plus_local_contrast_20epochs"), best("055_panel_scaled_contrast_two_stems_20epochs"))
    bars = axes[1].bar(names, values, color=("#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"))
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].set(ylabel="best water2 macro-IoU", title="Controlled representation comparison", ylim=(0, 0.8))
    axes[1].tick_params(axis="x", rotation=20)
    fig.savefig(CURRENT / "comparison_051_053_054_055.png", dpi=170)


if __name__ == "__main__":
    main()
