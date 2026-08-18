#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
exec /home/votter/miniconda3/envs/studcamp/bin/python "$dir/../001_mask_guided_gru/train.py" \
  --device cuda --temporal-model lstm --epochs 40 --batch-size 4 --lr 1e-3 \
  --weight-decay 1e-4 --hidden-size 64 --dropout 0.10 --seed 17 \
  --train-augmentation 012_geometric_affine --output-dir "$dir/artifacts"
