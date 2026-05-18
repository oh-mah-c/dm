#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/results/chuo_q1_grid}"
TIME_LIMIT="${CHUO_GRID_TIMEOUT:-60s}"

mkdir -p "$OUT_DIR"
make -C "$ROOT_DIR" bin/chuo_miner bin/dm.exe >/dev/null

append_run() {
    local algorithm="$1"
    local dataset="$2"
    local input="$3"
    local minutil="$4"
    local minsup="$5"
    local minocc="$6"
    local cmd="$7"
    local outfile="$OUT_DIR/${algorithm}_${dataset}.txt"

    {
        echo "===== algorithm=$algorithm dataset=$dataset minutil=$minutil minsup=$minsup minocc=$minocc ====="
        echo "command=$cmd"
        echo "timeout=$TIME_LIMIT"
    } >> "$outfile"

    if timeout "$TIME_LIMIT" bash -lc "$cmd" >> "$outfile" 2>&1; then
        echo "runner_status=OK" >> "$outfile"
    else
        local rc=$?
        if [ "$rc" -eq 124 ]; then
            echo "runner_status=TIMEOUT" >> "$outfile"
        else
            echo "runner_status=FAILED_$rc" >> "$outfile"
        fi
    fi
    echo >> "$outfile"
}

run_config() {
    local dataset="$1"
    local input="$2"
    local minutil="$3"
    local minsup="$4"
    local minocc="$5"

    append_run "chuo_miner" "$dataset" "$input" "$minutil" "$minsup" "$minocc" \
        "$ROOT_DIR/bin/chuo_miner --input $input --minutil $minutil --minsup $minsup --minocc $minocc"

    append_run "mhoui" "$dataset" "$input" "$minutil" "$minsup" "$minocc" \
        "$ROOT_DIR/bin/dm.exe mhoui $input 1 $minsup $minocc $minutil"

    append_run "chui_miner" "$dataset" "$input" "$minutil" "$minsup" "$minocc" \
        "$ROOT_DIR/bin/dm.exe chui_miner $input 1 $minutil"

    append_run "efim_closed" "$dataset" "$input" "$minutil" "$minsup" "$minocc" \
        "$ROOT_DIR/bin/dm.exe efim_closed $input 1 $minutil"

    append_run "efim" "$dataset" "$input" "$minutil" "$minsup" "$minocc" \
        "$ROOT_DIR/bin/dm.exe efim $input 1 $minutil"
}

run_dataset() {
    local dataset="$1"
    local input="$2"
    shift 2
    rm -f "$OUT_DIR/"*"_${dataset}.txt"
    while [ "$#" -gt 0 ]; do
        run_config "$dataset" "$input" "$1" "$2" "$3"
        shift 3
    done
}

run_dataset "foodmart" "$ROOT_DIR/datasets/utilities/foodmart.txt" \
    500 0.005 0.10 \
    750 0.006 0.15 \
    1000 0.0075 0.20

run_dataset "liquor_11" "$ROOT_DIR/datasets/utilities/liquor_11.txt" \
    2500 0.015 0.25 \
    5000 0.020 0.30 \
    10000 0.030 0.35

run_dataset "chainstore" "$ROOT_DIR/datasets/utilities/chainstore.txt" \
    25000 0.0075 0.15 \
    50000 0.0100 0.20 \
    100000 0.0150 0.25

MPLCONFIGDIR=/tmp/mpl python3 "$ROOT_DIR/scripts/plot_chuo_q1_grid.py" "$OUT_DIR"
