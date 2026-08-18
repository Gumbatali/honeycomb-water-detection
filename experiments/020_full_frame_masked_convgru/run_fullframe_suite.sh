#!/usr/bin/env bash
# Repeats experiments 010--018 with the full-frame 60-step ConvGRU.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="/home/votter/miniconda3/envs/studcamp/bin/python"
trainer="$root/experiments/020_full_frame_masked_convgru/train.py"

run() {
  local experiment="$1"
  shift
  local output="$root/experiments/$experiment"
  mkdir -p "$output"
  printf '%s\n' "Full-frame ConvGRU experiment" "Command: $python_bin $trainer $*" \
    "Input: full 240x320 panel, 60 frames at 0.5 s over 30 s, frozen U-Net soft mask." \
    "Validation: unaugmented water4 only." > "$output/RUN.txt"
  "$python_bin" "$trainer" --output-dir "$output/artifacts" "$@" | tee "$output/train.log"
}

run 021_fullframe_rotation_40 --epochs 40 --lr 3e-4 --train-augmentation 010_rotation
run 022_fullframe_flip_40 --epochs 40 --lr 3e-4 --train-augmentation 011_horizontal_flip
run 023_fullframe_affine_40 --epochs 40 --lr 3e-4 --train-augmentation 012_geometric_affine
run 024_fullframe_patching_40 --epochs 40 --lr 3e-4 --train-augmentation 013_background_patching
run 025_fullframe_shift_40 --epochs 40 --lr 3e-4 --train-augmentation 014_defect_location_shift
run 026_fullframe_aggressive_rotation_100 --epochs 100 --lr 3e-4 --train-augmentation 015_aggressive_rotation
run 027_fullframe_aggressive_affine_100 --epochs 100 --lr 3e-4 --train-augmentation 016_aggressive_affine
run 028_fullframe_affine_flip_patch_100 --epochs 100 --lr 1e-4 \
  --init-checkpoint "$root/experiments/027_fullframe_aggressive_affine_100/artifacts/best.pt" \
  --train-augmentation 016_aggressive_affine --train-augmentation 011_horizontal_flip --train-augmentation 013_background_patching
run 029_fullframe_all_augmentations_100 --epochs 100 --lr 1e-4 \
  --init-checkpoint "$root/experiments/028_fullframe_affine_flip_patch_100/artifacts/best.pt" \
  --train-augmentation 010_rotation --train-augmentation 011_horizontal_flip --train-augmentation 012_geometric_affine \
  --train-augmentation 013_background_patching --train-augmentation 014_defect_location_shift \
  --train-augmentation 015_aggressive_rotation --train-augmentation 016_aggressive_affine
