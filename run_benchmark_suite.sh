#!/usr/bin/env bash
set -euo pipefail

BIN="./build/bin/dm.exe"
OUTDIR="docs/core/AURA-HOI"
CSV="${OUTDIR}/benchmark_results.csv"
ITEMSET_CSV="${OUTDIR}/itemset_counts.csv"
TMP="$(mktemp)"
THREADS=$(nproc || echo "16")

echo "dataset,algorithm,alpha,minsup,time_s,peak_ram_mb,itemsets,visited_nodes" > "$CSV"

ceil_alpha_n() {
  local alpha="$1" n="$2"
  python3 - "$alpha" "$n" <<'PY'
import math, sys
alpha = float(sys.argv[1])
n = int(sys.argv[2])
print(max(1, math.ceil(alpha * n)))
PY
}

run_one() {
  local ds="$1" algo="$2" alpha="$3" minsup="$4"
  local dspath="${DATAFILES[$ds]}"
  echo "  → ${ds} | ${algo} | α=${alpha} | minsup=${minsup}"

  if [ "$algo" = "aura_hoi" ]; then
    timeout 120s /usr/bin/time -v "$BIN" aura_hoi "$dspath" 0 "$alpha" "$minsup" 0 sum raw -t "$THREADS" > "$TMP" 2>&1 || true
  else
    timeout 120s /usr/bin/time -v "$BIN" "$algo" "$dspath" 0 "$alpha" "$minsup" -t "$THREADS" > "$TMP" 2>&1 || true
  fi

  time_s=$(grep -oP "Algorithm Core\s*:\s*\K[0-9.]+" "$TMP" 2>/dev/null | head -1 || echo "")
  if [ -z "$time_s" ]; then
    time_s=$(grep -oP "TOTAL WALL TIME\s*:\s*\K[0-9.]+" "$TMP" 2>/dev/null | head -1 || echo "")
  fi
  if [ -z "$time_s" ]; then
    wall_raw=$(grep -oP "Elapsed \(wall clock\) time.*?:\s*\K[0-9:.]+" "$TMP" 2>/dev/null | head -1 || echo "")
    if [ -n "$wall_raw" ]; then
      time_s=$(python3 -c "
p='${wall_raw}'.split(':')
if len(p)==2: print(float(p[0])*60+float(p[1]))
elif len(p)==3: print(float(p[0])*3600+float(p[1])*60+float(p[2]))
else: print(p[0])
" 2>/dev/null || echo "")
    fi
  else
    time_s=$(python3 -c "print(${time_s}/1000.0)" 2>/dev/null || echo "$time_s")
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

  echo "${ds},${algo},${alpha},${minsup},${time_s:-},${ram_mb},${itemsets:-0},${v_nodes:-0}" >> "$CSV"
}

declare -A DATAFILES=(
  [mushrooms]="datasets/itemsets/mushrooms.txt"
  [chess]="datasets/itemsets/chess.txt"
  [retail]="datasets/itemsets/retail.txt"
  [T10I4D100K]="datasets/itemsets/T10I4D100K.txt"
  [kosarak]="datasets/itemsets/kosarak.dat"
  [pumsb]="datasets/itemsets/pumsb.txt"
)

declare -A NTRANS=(
  [mushrooms]=8416
  [chess]=3196
  [retail]=88162
  [T10I4D100K]=100000
  [kosarak]=990002
  [pumsb]=49046
)

declare -A ALPHAS=(
  [mushrooms]="0.05 0.075 0.10 0.125 0.15 0.20 0.30 0.40"
  [chess]="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85"
  [retail]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
  [T10I4D100K]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
  [kosarak]="0.001 0.002 0.005 0.01 0.02"
  [pumsb]="0.01 0.03 0.05 0.07 0.09"
)

ALGOS="hep dfhoi aura_hoi"

for ds in mushrooms chess retail T10I4D100K kosarak pumsb; do
  ntrans="${NTRANS[$ds]}"
  echo "▶ Dataset: $ds (n=${ntrans})"
  for alpha in ${ALPHAS[$ds]}; do
    minsup="$(ceil_alpha_n "$alpha" "$ntrans")"
    for algo in $ALGOS; do
      run_one "$ds" "$algo" "$alpha" "$minsup"
    done
  done
done

rm -f "$TMP"
echo "✓ Benchmark suite complete! Running plot_results.py..."
/home/hutech/self/cad_mcrs/.venv/bin/python3 docs/core/AURA-HOI/plot_results.py
echo "✓ Done."
