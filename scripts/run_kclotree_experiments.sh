#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-results/kclotree_compare_full}"
DATASET="datasets/Malware_types_in_SPMF_Format"
mkdir -p "$OUT_DIR"

make bin/kclotree_miner >/dev/null

OUT_FILE="$OUT_DIR/kclotree_malware_types.txt"
: > "$OUT_FILE"

for type in generic group redundancy_aware; do
  for k in 10 25; do
    {
      echo "=== kclotree dataset=malware_types type=$type k=$k ==="
      bin/kclotree_miner \
        --input "$DATASET" \
        --type "$type" \
        --k "$k" \
        --max-depth 5 \
        --max-candidates 200000 \
        --max-seconds 90
      echo
    } >> "$OUT_FILE"
  done
done

echo "Wrote $OUT_FILE"
