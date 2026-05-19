#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-results/tku_compare_full}"
mkdir -p "$OUT_DIR"

make bin/tku_miner >/dev/null

run_dataset() {
  local name="$1"
  local path="$2"
  local outfile="$OUT_DIR/tku_${name}.txt"
  : > "$outfile"
  for k in 10 50 100; do
    {
      echo "=== tku dataset=$name k=$k ==="
      bin/tku_miner --input "$path" --k "$k" --max-seconds 120
      echo
    } >> "$outfile"
  done
}

run_dataset "foodmart" "datasets/utilities/foodmart.txt"
run_dataset "liquor_11" "datasets/utilities/liquor_11.txt"

echo "Wrote TKU statistics to $OUT_DIR"
