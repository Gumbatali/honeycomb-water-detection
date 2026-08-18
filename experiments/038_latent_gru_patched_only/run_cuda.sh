#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
exec /home/votter/miniconda3/envs/studcamp/bin/python "$root/experiments/030_unet_latent_convgru/train.py" \
  --output-dir "$root/experiments/038_latent_gru_patched_only/artifacts" \
  --epochs 30 --hidden 128 --layers 1 --dropout 0 --lr 3e-5 --weight-decay 1e-4 \
  --num-classes 6 --ignore-label 6 --include-patched-only \
  --init-checkpoint "$root/experiments/037_latent_gru_no_water120_augmented/artifacts/best.pt"
