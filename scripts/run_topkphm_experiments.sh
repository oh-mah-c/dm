#!/usr/bin/env bash
set -u

OUT_DIR="${1:-results/topkphm_compare_full}"
mkdir -p "$OUT_DIR"

make bin/topkphm_miner >/dev/null

run_case() {
  local dataset="$1"
  local path="$2"
  local k="$3"
  local maxper="$4"
  local maxavg="$5"
  local depth="$6"
  local outfile="$OUT_DIR/topkphm_${dataset}.txt"
  {
    echo "=== topkphm dataset=$dataset k=$k maxper=$maxper maxavg=$maxavg ==="
    bin/topkphm_miner \
      --input "$path" \
      --k "$k" \
      --maxper "$maxper" \
      --maxavg "$maxavg" \
      --max-depth "$depth" \
      --max-candidates 1000000 \
      --max-seconds 180
    echo
  } >> "$outfile" 2>&1
}

: > "$OUT_DIR/topkphm_foodmart.txt"
: > "$OUT_DIR/topkphm_liquor_11.txt"
: > "$OUT_DIR/topkphm_fruithut.txt"
: > "$OUT_DIR/topkphm_chainstore.txt"

run_case foodmart datasets/utilities/foodmart.txt 50 500 500 5
run_case foodmart datasets/utilities/foodmart.txt 50 1000 1000 5
run_case foodmart datasets/utilities/foodmart.txt 100 1000 1000 5

run_case liquor_11 datasets/utilities/liquor_11.txt 50 200 100 5
run_case liquor_11 datasets/utilities/liquor_11.txt 100 500 250 5
run_case liquor_11 datasets/utilities/liquor_11.txt 100 1000 500 5

run_case fruithut datasets/utilities/fruithut_utility.txt 50 200 100 5
run_case fruithut datasets/utilities/fruithut_utility.txt 100 500 250 5
run_case fruithut datasets/utilities/fruithut_utility.txt 100 1000 500 5

run_case chainstore datasets/utilities/chainstore.txt 50 1000 500 4
run_case chainstore datasets/utilities/chainstore.txt 100 2000 1000 4

echo "TOPKPHM statistics written to $OUT_DIR"
