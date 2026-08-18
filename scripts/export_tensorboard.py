#!/usr/bin/env python3
"""Export LSTM metrics and augmentation QA diagrams into one TensorBoard logdir."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def add_training_runs(writer: SummaryWriter) -> None:
    for metrics_path in sorted(EXPERIMENTS.glob("*/artifacts/metrics.json")):
        name = metrics_path.parents[1].name
        report = json.loads(metrics_path.read_text())
        for row in report.get("history", []):
            epoch = int(row["epoch"])
            for key in row:
                if key == "epoch":
                    continue
                writer.add_scalar(f"{name}/validation/{key}", float(row[key]), epoch)
        best = report.get("best_validation", {})
        writer.add_text(f"{name}/run_summary", "<br>".join(f"{key}: {value}" for key, value in best.items()))


def add_augmentation_plots(writer: SummaryWriter) -> None:
    for directory in sorted(EXPERIMENTS.glob("01[0-9]_*")):
        readme = directory / "README.txt"
        if readme.exists():
            writer.add_text(f"{directory.name}/description", readme.read_text().replace("\n", "<br>"))
        image_paths = list((directory / "plots").glob("*.png")) + list((directory / "artifacts").glob("*.png"))
        for image_path in sorted(image_paths):
            image = np.asarray(Image.open(image_path).convert("RGB")).transpose(2, 0, 1)
            writer.add_image(f"{directory.name}/qa/{image_path.stem}", image, global_step=0, dataformats="CHW")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default=EXPERIMENTS / "tensorboard_logs")
    parser.add_argument("--clean", action="store_true", help="replace the generated log directory")
    args = parser.parse_args()
    logdir = args.logdir.resolve()
    if args.clean and logdir.exists():
        shutil.rmtree(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)
    add_training_runs(writer)
    add_augmentation_plots(writer)
    writer.flush()
    writer.close()
    print(f"TensorBoard log written to {logdir}")


if __name__ == "__main__":
    main()
