#!/usr/bin/env bash
set -euo pipefail

CSV_ST="docs/core/AURA-HOI/benchmark_results_single_thread.csv"
CSV_MT="docs/core/AURA-HOI/benchmark_results.csv"
TMP="$(mktemp)"
BIN="./build/bin/dm.exe"

echo "dataset,algorithm,alpha,minsup,time_s,peak_ram_mb,itemsets,visited_nodes" > "$CSV_ST"
echo "dataset,algorithm,alpha,minsup,time_s,peak_ram_mb,itemsets,visited_nodes" > "$CSV_MT"

declare -A DATAFILES=(
  [mushrooms]="datasets/itemsets/mushrooms.txt"
  [chess]="datasets/itemsets/chess.txt"
  [retail]="datasets/itemsets/retail.txt"
  [T10I4D100K]="datasets/itemsets/T10I4D100K.txt"
  [kosarak]="datasets/itemsets/kosarak.dat"
)

declare -A NTRANS=(
  [mushrooms]=8416
  [chess]=3196
  [retail]=88162
  [T10I4D100K]=100000
  [kosarak]=990002
)

declare -A ALPHAS=(
  [mushrooms]="0.05 0.075 0.10 0.125 0.15 0.20 0.30 0.40"
  [chess]="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85"
  [retail]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
  [T10I4D100K]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
  [kosarak]="0.001 0.002 0.005 0.01 0.02"
)

ceil_alpha_n() {
  python3 -c "import math; print(max(1, math.ceil($1 * $2)))"
}

run_bench() {
  local ds="$1" algo="$2" alpha="$3" minsup="$4" threads="$5" target_csv="$6"
  local dspath="${DATAFILES[$ds]}"
  echo "  [${threads}T] ${ds} | ${algo} | α=${alpha}"

  if [ "$algo" = "aura_hoi" ]; then
    timeout 120s /usr/bin/time -v "$BIN" aura_hoi "$dspath" 0 "$alpha" "$minsup" 0 sum raw -t "$threads" > "$TMP" 2>&1 || true
  else
    timeout 120s /usr/bin/time -v "$BIN" "$algo" "$dspath" 0 "$alpha" "$minsup" -t "$threads" > "$TMP" 2>&1 || true
  fi

  time_s=$(grep -oP "Algorithm Core\s*:\s*\K[0-9.]+" "$TMP" 2>/dev/null | head -1 || echo "")
  if [ -z "$time_s" ]; then
    time_s=$(grep -oP "TOTAL WALL TIME\s*:\s*\K[0-9.]+" "$TMP" 2>/dev/null | head -1 || echo "")
  fi
  if [ -n "$time_s" ]; then
    time_s=$(python3 -c "print(${time_s}/1000.0)" 2>/dev/null || echo "$time_s")
  else
    time_s=""
  fi

  ram_kb=$(grep -oP "Peak RAM \(VmHWM\)\s*:\s*[0-9.]+\s*MB\s*\(\K\d+" "$TMP" 2>/dev/null | head -1 || echo "")
  if [ -z "$ram_kb" ]; then
    ram_kb=$(grep -oP "Maximum resident set size \(kbytes\):\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "0")
  fi
  ram_mb=$(python3 -c "print(${ram_kb:-0}/1024.0)" 2>/dev/null || echo "0")

  itemsets=$(grep -oP "Frequent Itemsets\s*:\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "")
  if [ -z "$itemsets" ]; then
    itemsets=$(grep -oiP "(?:high occupancy )?itemsets found:\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "0")
  fi

  v_nodes=$(grep -oP "visited_nodes=\K\d+" "$TMP" 2>/dev/null | head -1 || echo "0")

  echo "${ds},${algo},${alpha},${minsup},${time_s:-},${ram_mb},${itemsets:-0},${v_nodes:-0}" >> "$target_csv"
}

# 1. Run Single-threaded benchmark (-t 1)
echo "=== RUNNING SINGLE-THREADED BENCHMARK (-t 1) ==="
for ds in mushrooms chess retail T10I4D100K kosarak; do
  ntrans="${NTRANS[$ds]}"
  for alpha in ${ALPHAS[$ds]}; do
    minsup=$(ceil_alpha_n "$alpha" "$ntrans")
    for algo in hep dfhoi aura_hoi; do
      run_bench "$ds" "$algo" "$alpha" "$minsup" 1 "$CSV_ST"
    done
  done
done

# 2. Run Multi-threaded benchmark (-t 16)
echo "=== RUNNING MULTI-THREADED BENCHMARK (-t 16) ==="
for ds in mushrooms chess retail T10I4D100K kosarak; do
  ntrans="${NTRANS[$ds]}"
  for alpha in ${ALPHAS[$ds]}; do
    minsup=$(ceil_alpha_n "$alpha" "$ntrans")
    for algo in hep dfhoi aura_hoi; do
      run_bench "$ds" "$algo" "$alpha" "$minsup" 16 "$CSV_MT"
    done
  done
done

rm -f "$TMP"
echo "✓ Benchmark execution finished!"
