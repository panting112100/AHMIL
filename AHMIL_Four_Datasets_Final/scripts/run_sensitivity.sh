#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_sensitivity.sh DATASET DATA_DIR TASK [OUTPUT_DIR] [extra train.py args...]"
  exit 1
fi
DATASET="$1"; DATA_DIR="$2"; TASK="$3"; OUTPUT_DIR="${4:-./outputs/sensitivity}"
if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi
for RULES in 3 5 7 9 11; do
  python -m scripts.train --dataset "$DATASET" --data_dir "$DATA_DIR" --task "$TASK" \
    --variant full --fuzzynum "$RULES" --validation_only --output_dir "$OUTPUT_DIR/fuzzynum_${RULES}" "$@"
done
for CURVATURE in 0.5 1.0 1.5 2.0 2.5; do
  python -m scripts.train --dataset "$DATASET" --data_dir "$DATA_DIR" --task "$TASK" \
    --variant full --curvature "$CURVATURE" --validation_only --output_dir "$OUTPUT_DIR/curvature_${CURVATURE}" "$@"
done
