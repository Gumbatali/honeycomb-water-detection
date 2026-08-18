#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/home/votter/miniconda3/envs/studcamp/bin/python"
"$python_bin" "$root/scripts/export_tensorboard.py" --clean
exec "$python_bin" -m tensorboard.main --logdir "$root/experiments/tensorboard_logs" --host 127.0.0.1 --port 6006
