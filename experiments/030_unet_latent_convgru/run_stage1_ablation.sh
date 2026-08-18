#!/usr/bin/env bash
# Feature-fusion ablation.  water1+augmentations tunes, water2 validates.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="/home/votter/miniconda3/envs/studcamp/bin/python"
trainer="$root/experiments/030_unet_latent_convgru/train.py"

run() {
  local name="$1"
  shift
  local out="$root/experiments/$name"
  mkdir -p "$out"
  printf '%s\n' "Experiment $name" "Frozen U-Net v1 latent features; no U-Net mask input." \
    "Tuning split: train=water1+all augmentations, validation=water2; water4 untouched." \
    "Command: $python_bin $trainer $*" > "$out/RUN.txt"
  "$python_bin" "$trainer" --output-dir "$out/artifacts" "$@" | tee "$out/train.log"
}

# Same temporal capacity and optimisation; only U-Net feature fusion differs.
run 031_latent_e4_only --epochs 50 --hidden 160 --layers 2 --dropout .10 --lr 2e-4 --weight-decay 1e-4
run 032_latent_e4_d3 --epochs 50 --hidden 160 --layers 2 --dropout .10 --lr 2e-4 --weight-decay 1e-4 --use-d3
