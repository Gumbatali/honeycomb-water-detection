#!/usr/bin/env bash
# Stage 2: GRU capacity and optimiser search with e4-only fusion.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="/home/votter/miniconda3/envs/studcamp/bin/python"
trainer="$root/experiments/030_unet_latent_convgru/train.py"

run() {
  local name="$1"
  shift
  local out="$root/experiments/$name"
  mkdir -p "$out"
  printf '%s\n' "Experiment $name" "Frozen U-Net v1 e4-only features; no mask, no d3." \
    "Tuning split: train=water1+all augmentations, validation=water2; water4 untouched." \
    "Command: $python_bin $trainer $*" > "$out/RUN.txt"
  "$python_bin" "$trainer" --output-dir "$out/artifacts" "$@" | tee "$out/train.log"
}

# C is experiment 031 and is reused rather than rerun.
run 033_latent_gru_1x128 --epochs 50 --hidden 128 --layers 1 --dropout 0 --lr 3e-4 --weight-decay 1e-4
run 034_latent_gru_1x160 --epochs 50 --hidden 160 --layers 1 --dropout 0 --lr 2e-4 --weight-decay 1e-4
run 035_latent_gru_2x192 --epochs 50 --hidden 192 --layers 2 --dropout .10 --lr 1e-4 --weight-decay 5e-4
