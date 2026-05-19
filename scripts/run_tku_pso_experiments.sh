#!/usr/bin/env bash
set -u

OUT_DIR="${1:-results/tku_pso_compare_full}"
mkdir -p "$OUT_DIR"

make bin/tku_pso_miner >/dev/null

run_dataset() {
  local name="$1"
  local path="$2"
  local iterations="$3"
  local outfile="$OUT_DIR/tku_pso_${name}.txt"
  : > "$outfile"
  for k in 50 100 500; do
    {
      echo "=== tku_pso dataset=$name k=$k population=20 iterations=$iterations ==="
      bin/tku_pso_miner \
        --input "$path" \
        --k "$k" \
        --population 20 \
        --iterations "$iterations" \
        --seed 42 \
        --max-seconds 180
      echo
    } >> "$outfile" 2>&1
  done
}

run_dataset foodmart datasets/utilities/foodmart.txt 1000
run_dataset liquor_11 datasets/utilities/liquor_11.txt 500
run_dataset fruithut datasets/utilities/fruithut_utility.txt 300

echo "TKU-PSO statistics written to $OUT_DIR"
