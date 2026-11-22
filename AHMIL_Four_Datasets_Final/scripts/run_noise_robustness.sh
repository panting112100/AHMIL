#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_noise_robustness.sh DATASET DATA_DIR TASK [OUTPUT_DIR] [extra train.py args...]"
  exit 1
fi
DATASET="$1"; DATA_DIR="$2"; TASK="$3"; OUTPUT_DIR="${4:-./outputs/noise_robustness}"
if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi
for RATIO in 0 0.25 0.5 0.75; do
  for VARIANT in baseline fuzzy; do
    python -m scripts.train --dataset "$DATASET" --data_dir "$DATA_DIR" --task "$TASK" \
      --variant "$VARIANT" --interference_ratio "$RATIO" \
      --output_dir "$OUTPUT_DIR/ratio_${RATIO}/$VARIANT" "$@"
  done
done
