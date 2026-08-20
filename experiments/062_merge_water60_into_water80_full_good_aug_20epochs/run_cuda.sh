#!/usr/bin/env bash
set -euo pipefail
root="/home/votter/projects/honeycomb-water-detection"
exec conda run --no-capture-output -n studcamp python \
  "$root/experiments/030_unet_feature_pyramid_convgru/train.py" \
  --output-dir "$root/experiments/062_merge_water60_into_water80_full_good_aug_20epochs/artifacts" \
  --unet-checkpoint "$root/models/segmentation/v1/unet_water_v2.pth" \
  --train-video water1 --valid-video water2 \
  --online-domain-augmentation --online-temporal-augmentation --online-repeats 16 \
  --include-patched-only --include-cell-permutation-gain \
  --num-classes 5 --merge-label 3 --merge-into 4 --compact-after-merge --merged-class-weight 0.5 \
  --ignore-label 6 --neutralize-label 6 \
  --apply-roi --thermal-normalization pixel_peak \
  --amp-dtype bf16 --hidden 128 --dropout 0.20 \
  --lr 1e-4 --weight-decay 1e-4 --epochs 20 --patience 30 --seed 17
