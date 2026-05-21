#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/results/hupp_compare_full}"
BIN="$ROOT_DIR/bin/hupp_miner"

mkdir -p "$OUT_DIR"
make -C "$ROOT_DIR" bin/hupp_miner >/dev/null

run_case() {
    local dataset_name="$1"
    local dataset_type="$2"
    local input="$3"
    shift 3
    local outfile="$OUT_DIR/hupp_${dataset_name}.txt"
    : > "$outfile"
    for cfg in "$@"; do
        echo "===== $cfg =====" >> "$outfile"
        "$BIN" --input "$input" --dataset-type "$dataset_type" $cfg >> "$outfile"
        echo >> "$outfile"
    done
}

run_case "databricks_dolly_15k" "dolly" "$ROOT_DIR/datasets/prompt/databricks-dolly-15k.jsonl" \
    "--minsup 0.02 --minutil-ratio 0.002 --minalign 0.80 --max-depth 6" \
    "--minsup 0.03 --minutil-ratio 0.003 --minalign 0.82 --max-depth 6" \
    "--minsup 0.05 --minutil-ratio 0.005 --minalign 0.85 --max-depth 5"

run_case "modified_code_feedback" "code_feedback" "$ROOT_DIR/datasets/prompt/Modified-Code-Feedback.jsonl" \
    "--minsup 0.02 --minutil-ratio 0.002 --minalign 0.80 --max-depth 5" \
    "--minsup 0.03 --minutil-ratio 0.003 --minalign 0.82 --max-depth 5" \
    "--minsup 0.05 --minutil-ratio 0.005 --minalign 0.85 --max-depth 4"

MPLCONFIGDIR=/tmp/mpl python3 "$ROOT_DIR/scripts/plot_hupp_results.py" "$OUT_DIR"
