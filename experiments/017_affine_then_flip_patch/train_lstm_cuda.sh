#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
exec /home/votter/miniconda3/envs/studcamp/bin/python "$dir/../001_mask_guided_gru/train.py" \
  --device cuda --temporal-model lstm --epochs 100 --batch-size 4 --lr 5e-4 \
  --weight-decay 1e-4 --hidden-size 64 --dropout 0.10 --seed 17 \
  --init-checkpoint "$dir/../016_aggressive_affine/artifacts/best.pt" \
  --train-augmentation 016_aggressive_affine --train-augmentation 011_horizontal_flip \
  --train-augmentation 013_background_patching --output-dir "$dir/artifacts"
