#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
exec /home/votter/miniconda3/envs/studcamp/bin/python "$dir/train.py" --output-dir "$dir/artifacts" "$@"
