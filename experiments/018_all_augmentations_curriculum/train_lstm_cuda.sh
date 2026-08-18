#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
exec /home/votter/miniconda3/envs/studcamp/bin/python "$dir/../001_mask_guided_gru/train.py" \
  --device cuda --temporal-model lstm --epochs 100 --batch-size 8 --lr 3e-4 \
  --weight-decay 1e-4 --hidden-size 64 --dropout 0.10 --seed 17 \
  --init-checkpoint "$dir/../017_affine_then_flip_patch/artifacts/best.pt" \
  --train-augmentation 010_rotation --train-augmentation 011_horizontal_flip \
  --train-augmentation 012_geometric_affine --train-augmentation 013_background_patching \
  --train-augmentation 014_defect_location_shift --train-augmentation 015_aggressive_rotation \
  --train-augmentation 016_aggressive_affine --output-dir "$dir/artifacts"
