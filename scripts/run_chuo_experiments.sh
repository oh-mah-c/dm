#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/results/chuo_compare_full}"
BIN="$ROOT_DIR/bin/chuo_miner"

mkdir -p "$OUT_DIR"
make -C "$ROOT_DIR" bin/chuo_miner >/dev/null

run_case() {
    local dataset_name="$1"
    local input="$2"
    shift 2
    local outfile="$OUT_DIR/chuo_${dataset_name}.txt"
    : > "$outfile"
    for cfg in "$@"; do
        echo "===== $cfg =====" >> "$outfile"
        "$BIN" --input "$input" $cfg >> "$outfile"
        echo >> "$outfile"
    done
}

run_case "foodmart" "$ROOT_DIR/datasets/utilities/foodmart.txt" \
    "--minutil 500 --minsup 0.005 --minocc 0.10 --max-depth 6" \
    "--minutil 1000 --minsup 0.0075 --minocc 0.15 --max-depth 6" \
    "--minutil 2000 --minsup 0.01 --minocc 0.20 --max-depth 5"

run_case "liquor_11" "$ROOT_DIR/datasets/utilities/liquor_11.txt" \
    "--minutil 1000 --minsup 0.01 --minocc 0.20 --max-depth 5" \
    "--minutil 2500 --minsup 0.015 --minocc 0.25 --max-depth 5" \
    "--minutil 5000 --minsup 0.02 --minocc 0.30 --max-depth 4"

run_case "chainstore" "$ROOT_DIR/datasets/utilities/chainstore.txt" \
    "--minutil 10000 --minsup 0.005 --minocc 0.10 --max-depth 5" \
    "--minutil 25000 --minsup 0.0075 --minocc 0.15 --max-depth 5" \
    "--minutil 50000 --minsup 0.01 --minocc 0.20 --max-depth 4"

MPLCONFIGDIR=/tmp/mpl python3 "$ROOT_DIR/scripts/plot_chuo_results.py" "$OUT_DIR"
