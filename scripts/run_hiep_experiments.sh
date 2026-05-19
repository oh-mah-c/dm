#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/results/hiep_q1}"
BIN="$ROOT_DIR/bin/dm.exe"

mkdir -p "$OUT_DIR"
make -C "$ROOT_DIR" bin/dm.exe >/dev/null
python3 "$ROOT_DIR/scripts/run_hiep_experiments.py" "$OUT_DIR" "$BIN" "$ROOT_DIR"
MPLCONFIGDIR=/tmp/mpl python3 "$ROOT_DIR/scripts/plot_hiep_results.py" "$OUT_DIR"
python3 "$ROOT_DIR/scripts/run_hiep_baseline_experiments.py" "$OUT_DIR" "$BIN" "$ROOT_DIR" 35
