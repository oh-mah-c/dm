#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-results/pso_classifier_full}"
DATASET="datasets/Malware_types_in_SPMF_Format"
mkdir -p "$OUT_DIR"

if [[ ! -d "$DATASET" ]]; then
  echo "Missing dataset folder: $DATASET" >&2
  exit 1
fi

make bin/pso_classifier >/dev/null

OUT_FILE="$OUT_DIR/pso_classifier_malware_types.txt"
: > "$OUT_FILE"

CONFIGS=(
  "25 0.85 0.02"
  "25 0.90 0.02"
  "25 0.95 0.02"
  "50 0.90 0.02"
)

for cfg in "${CONFIGS[@]}"; do
  read -r particles threshold radius <<< "$cfg"
  {
    echo "=== pso_classifier malware_types particles=$particles threshold=$threshold radius=$radius ==="
    bin/pso_classifier \
      --input "$DATASET" \
      --particles "$particles" \
      --threshold "$threshold" \
      --radius "$radius" \
      --uncovered 0.10 \
      --max-iterations 300 \
      --seed 7
    echo
  } >> "$OUT_FILE"
done

echo "Wrote $OUT_FILE"
