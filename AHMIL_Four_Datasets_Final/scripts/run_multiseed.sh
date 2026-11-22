#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_multiseed.sh DATASET DATA_DIR TASK [OUTPUT_DIR] [extra train.py args...]"
  exit 1
fi
DATASET="$1"; DATA_DIR="$2"; TASK="$3"; OUTPUT_DIR="${4:-./outputs/multiseed}"
if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi
for SEED in 42 123 2026; do
  python -m scripts.train \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --task "$TASK" \
    --seed "$SEED" --output_dir "$OUTPUT_DIR/seed_${SEED}" "$@"
done
