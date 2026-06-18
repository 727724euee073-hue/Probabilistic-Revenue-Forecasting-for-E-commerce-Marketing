#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

python -m src.generate_features --data-dir "$DATA_DIR" --output-dir "$(dirname "$OUTPUT_PATH")"
python -m src.predict --features "$(dirname "$OUTPUT_PATH")/features.csv" --model "$MODEL_PATH" --output "$OUTPUT_PATH"
