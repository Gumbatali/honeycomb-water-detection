#!/usr/bin/env bash
set -euo pipefail
root="/home/votter/projects/honeycomb-water-detection"
exec conda run --no-capture-output -n studcamp python "$root/experiments/056_segmentation_v2_evaluation/evaluate.py"
