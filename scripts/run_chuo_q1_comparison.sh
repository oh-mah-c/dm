#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/results/chuo_q1_comparison}"
TIME_LIMIT="${CHUO_BASELINE_TIMEOUT:-30s}"

mkdir -p "$OUT_DIR/logs"
make -C "$ROOT_DIR" bin/chuo_miner bin/dm.exe >/dev/null

run_cmd() {
    local label="$1"
    local dataset="$2"
    local setting="$3"
    local cmd="$4"
    local logfile="$OUT_DIR/logs/${label}_${dataset}_${setting}.log"
    {
        echo "label=$label"
        echo "dataset=$dataset"
        echo "setting=$setting"
        echo "command=$cmd"
        echo "timeout=$TIME_LIMIT"
        echo "----- output -----"
    } > "$logfile"
    if timeout "$TIME_LIMIT" bash -lc "$cmd" >> "$logfile" 2>&1; then
        echo "runner_status=OK" >> "$logfile"
    else
        rc=$?
        if [ "$rc" -eq 124 ]; then
            echo "runner_status=TIMEOUT" >> "$logfile"
        else
            echo "runner_status=FAILED_$rc" >> "$logfile"
        fi
    fi
}

run_setting() {
    local dataset="$1"
    local input="$2"
    local setting="$3"
    local minutil="$4"
    local minsup="$5"
    local minocc="$6"

    run_cmd "chuo_miner" "$dataset" "$setting" \
        "$ROOT_DIR/bin/chuo_miner --input $input --minutil $minutil --minsup $minsup --minocc $minocc"

    run_cmd "mhoui" "$dataset" "$setting" \
        "$ROOT_DIR/bin/dm.exe mhoui $input 1 $minsup $minocc $minutil"

    run_cmd "chui_miner" "$dataset" "$setting" \
        "$ROOT_DIR/bin/dm.exe chui_miner $input 1 $minutil"

    run_cmd "efim_closed" "$dataset" "$setting" \
        "$ROOT_DIR/bin/dm.exe efim_closed $input 1 $minutil"

    run_cmd "efim" "$dataset" "$setting" \
        "$ROOT_DIR/bin/dm.exe efim $input 1 $minutil"
}

run_dataset_grid() {
    local dataset="$1"
    local input="$2"
    shift 2
    while [ "$#" -gt 0 ]; do
        run_setting "$dataset" "$input" "$1" "$2" "$3" "$4"
        shift 4
    done
}

run_dataset_grid "foodmart" "$ROOT_DIR/datasets/utilities/foodmart.txt" \
    "u500_s0p005_o0p10" 500 0.005 0.10 \
    "u500_s0p0075_o0p15" 500 0.0075 0.15 \
    "u1000_s0p005_o0p20" 1000 0.005 0.20 \
    "u1000_s0p01_o0p15" 1000 0.01 0.15 \
    "u2000_s0p01_o0p20" 2000 0.01 0.20

run_dataset_grid "liquor_11" "$ROOT_DIR/datasets/utilities/liquor_11.txt" \
    "u1000_s0p01_o0p20" 1000 0.01 0.20 \
    "u2500_s0p015_o0p25" 2500 0.015 0.25 \
    "u5000_s0p02_o0p30" 5000 0.02 0.30 \
    "u7500_s0p025_o0p35" 7500 0.025 0.35 \
    "u10000_s0p03_o0p40" 10000 0.03 0.40

run_dataset_grid "chainstore" "$ROOT_DIR/datasets/utilities/chainstore.txt" \
    "u10000_s0p005_o0p10" 10000 0.005 0.10 \
    "u25000_s0p0075_o0p15" 25000 0.0075 0.15 \
    "u50000_s0p01_o0p20" 50000 0.01 0.20 \
    "u75000_s0p0125_o0p25" 75000 0.0125 0.25 \
    "u100000_s0p015_o0p30" 100000 0.015 0.30

MPLCONFIGDIR=/tmp/mpl python3 "$ROOT_DIR/scripts/plot_chuo_q1_comparison.py" "$OUT_DIR"
