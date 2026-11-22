#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_ablation.sh DATASET DATA_DIR TASK [OUTPUT_DIR] [extra train.py args...]"
  exit 1
fi
DATASET="$1"; DATA_DIR="$2"; TASK="$3"; OUTPUT_DIR="${4:-./outputs/ablation}"
if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi
for VARIANT in baseline mlp fuzzy hyper fuzzy_epc full; do
  python -m scripts.train \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --task "$TASK" \
    --variant "$VARIANT" --output_dir "$OUTPUT_DIR/$VARIANT" "$@"
done
