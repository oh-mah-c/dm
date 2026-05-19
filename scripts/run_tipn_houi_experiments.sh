#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-results/tipn_houi_compare_full}"
mkdir -p "$OUT_DIR"

make bin/tipn_houi_miner >/dev/null

run_one() {
  local name="$1"
  local path="$2"
  local outfile="$OUT_DIR/tipn_houi_${name}.txt"
  : > "$outfile"
  for k in 50 100 150; do
    {
      echo "=== tipn_houi dataset=$name k=$k intervals=5 ==="
      bin/tipn_houi_miner \
        --input "$path" \
        --k "$k" \
        --intervals 5 \
        --max-depth 3 \
        --max-transactions 5000 \
        --max-items 200 \
        --max-candidates 50000 \
        --max-seconds 30 || true
      echo
    } >> "$outfile"
  done
}

run_one "retail_negative" "datasets/negative_unit_profit/retail_negative.txt"
run_one "mushroom_negative" "datasets/negative_unit_profit/mushroom_negative.txt"

echo "Wrote TIPN-HOUI statistics to $OUT_DIR"
