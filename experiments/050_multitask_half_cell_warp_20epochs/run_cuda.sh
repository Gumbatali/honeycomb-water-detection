#!/usr/bin/env bash
set -euo pipefail
root="/home/votter/projects/honeycomb-water-detection"
exec conda run --no-capture-output -n studcamp python \
  "$root/experiments/030_unet_feature_pyramid_convgru/train.py" \
  --output-dir "$root/experiments/050_multitask_half_cell_warp_20epochs/artifacts" \
  --unet-checkpoint "$root/models/segmentation/v1/unet_water_v2.pth" \
  --train-video water1 --valid-video water2 \
  --online-domain-augmentation --online-temporal-augmentation \
  --online-cell-temporal-augmentation \
  --cell-temporal-shift 0.125 --cell-temporal-time-delta 0.025 \
  --cell-temporal-cooling-delta 0.05 --online-repeats 16 \
  --num-classes 6 --merge-label 6 --merge-into 5 --merged-class-weight 0.5 \
  --multitask-heads --binary-loss-weight 0.30 --ordinal-loss-weight 0.20 \
  --apply-roi --thermal-normalization pixel_peak \
  --amp-dtype bf16 --hidden 128 --dropout 0.20 \
  --lr 1e-4 --weight-decay 1e-4 --epochs 20 --patience 30 --seed 17
