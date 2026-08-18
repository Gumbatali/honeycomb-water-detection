#!/usr/bin/env bash
set -euo pipefail
root="/home/votter/projects/honeycomb-water-detection"
exec conda run --no-capture-output -n studcamp python \
  "$root/experiments/030_unet_feature_pyramid_convgru/train.py" \
  --output-dir "$root/experiments/039_unet_v1_pyramid_gru_seed17/artifacts" \
  --unet-checkpoint "$root/models/segmentation/v1/unet_water_v2.pth" \
  --train-video water1 --valid-video water2 \
  --include-patched-only --num-classes 6 --ignore-label 6 \
  --amp-dtype bf16 \
  --hidden 128 --dropout 0.20 --lr 1e-4 --weight-decay 1e-4 \
  --epochs 100 --patience 30 --seed 17
