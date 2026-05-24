#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# AURA-HOI Benchmark — runs all algorithms, collects metrics, plots charts
# Usage: ./docs/core/AURA-HOI/benchmark.sh
# Outputs: docs/core/AURA-HOI/benchmark_results.csv
#          docs/core/AURA-HOI/itemset_counts.csv
#          docs/core/AURA-HOI/<dataset>_comparison.png
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BIN="./build/bin/dm.exe"
OUTDIR="docs/core/AURA-HOI"
CSV="${OUTDIR}/benchmark_results.csv"
ITEMSET_CSV="${OUTDIR}/itemset_counts.csv"
TMP="$(mktemp)"

# ── Init CSV ────────────────────────────────────────────────────────────
echo "dataset,algorithm,alpha,minsup,time_s,peak_ram_mb,itemsets" > "$CSV"

ceil_alpha_n() {
  local alpha="$1" n="$2"
  python3 - "$alpha" "$n" <<'PY'
import math
import sys

alpha = float(sys.argv[1])
n = int(sys.argv[2])
print(max(1, math.ceil(alpha * n)))
PY
}

# ── run_one: execute one (dataset, algo, alpha) and append a CSV row ────
run_one() {
  local ds="$1" algo="$2" alpha="$3" minsup="$4"
  local dspath="${DATAFILES[$ds]}"
  echo "  → ${ds} | ${algo} | α=${alpha}"

  if [ "$algo" = "aura_hoi" ]; then
    /usr/bin/time -v "$BIN" aura_hoi "$dspath" 0 "$alpha" "$minsup" 0 sum raw > "$TMP" 2>&1 || true
  else
    /usr/bin/time -v "$BIN" "$algo" "$dspath" 0 "$alpha" "$minsup" > "$TMP" 2>&1 || true
  fi

  # ── Parse metrics ──────────────────────────────────────────────────
  time_s=$(grep -oP "TOTAL WALL TIME\s*:\s*\K[0-9.]+" "$TMP" 2>/dev/null | head -1 || echo "")
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
    # convert ms → s
    time_s=$(python3 -c "print(${time_s}/1000.0)" 2>/dev/null || echo "$time_s")
  fi

  ram_kb=$(grep -oP "Maximum resident set size \(kbytes\):\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "0")
  ram_mb=$(python3 -c "print(${ram_kb:-0}/1024.0)" 2>/dev/null || echo "0")

  itemsets=$(grep -oP "Frequent Itemsets\s*:\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "")
  if [ -z "$itemsets" ]; then
    itemsets=$(grep -oiP "(?:high occupancy )?itemsets found:\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "0")
  fi

  echo "${ds},${algo},${alpha},${minsup},${time_s:-},${ram_mb},${itemsets:-0}" >> "$CSV"
}

checked_ntrans() {
  local ds="$1"
  local dspath="${DATAFILES[$ds]}"
  local expected="${NTRANS[$ds]}"
  local actual
  actual="$(awk 'END{print NR}' "$dspath")"
  if [ "$actual" != "$expected" ]; then
    echo "warning: ${ds} NTRANS map says ${expected}, file has ${actual}; using file count" >&2
    echo "$actual"
  else
    echo "$expected"
  fi
}

# ── Threshold matrix ────────────────────────────────────────────────────
declare -A NTRANS=(
  [mushrooms]=8416
  [chess]=3196
  [retail]=88162
  [T10I4D100K]=100000
  [kosarak]=990002
)
declare -A DATAFILES=(
  [mushrooms]="datasets/itemsets/mushrooms.txt"
  [chess]="datasets/itemsets/chess.txt"
  [retail]="datasets/itemsets/retail.txt"
  [T10I4D100K]="datasets/itemsets/T10I4D100K.txt"
  [kosarak]="datasets/itemsets/kosarak.dat"
)
declare -A ALPHAS=(
  [mushrooms]="0.05 0.075 0.10 0.125 0.15 0.20 0.30 0.40"
  [chess]="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85"
  [retail]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
  [T10I4D100K]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
  [kosarak]="0.001 0.002 0.005 0.01 0.02"
)

ALGOS="hep dfhoi aura_hoi"

# ── Run all configurations ──────────────────────────────────────────────
for ds in mushrooms chess retail T10I4D100K kosarak; do
  ntrans="$(checked_ntrans "$ds")"
  echo "▶ Dataset: $ds (n=${ntrans})"
  for alpha in ${ALPHAS[$ds]}; do
    minsup="$(ceil_alpha_n "$alpha" "$ntrans")"
    for algo in $ALGOS; do
      run_one "$ds" "$algo" "$alpha" "$minsup"
    done
  done
done

rm -f "$TMP"
echo ""
echo "✓ Results saved → ${CSV}"

# ── Generate itemset-count CSV and charts with inline Python ───────────
VENV_PY="./.venv/bin/python3"
[ -x "$VENV_PY" ] || VENV_PY="python3"

$VENV_PY - <<'PYEOF'
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV  = Path("docs/core/AURA-HOI/benchmark_results.csv")
OUT  = Path("docs/core/AURA-HOI")
ITEMSET_CSV = OUT / "itemset_counts.csv"

df = pd.read_csv(CSV)
df["time_s"]      = pd.to_numeric(df["time_s"],      errors="coerce")
df["peak_ram_mb"] = pd.to_numeric(df["peak_ram_mb"], errors="coerce")
df["itemsets"]    = pd.to_numeric(df["itemsets"],     errors="coerce")

itemset_stats = (
    df.pivot_table(
        index=["dataset", "alpha", "minsup"],
        columns="algorithm",
        values="itemsets",
        aggfunc="first",
    )
    .reset_index()
)
for col in ["hep", "dfhoi", "aura_hoi"]:
    if col not in itemset_stats.columns:
        itemset_stats[col] = pd.NA
itemset_stats = itemset_stats[["dataset", "alpha", "minsup", "hep", "dfhoi", "aura_hoi"]]
itemset_stats = itemset_stats.rename(columns={
    "hep": "hep_itemsets",
    "dfhoi": "dfhoi_itemsets",
    "aura_hoi": "aura_hoi_itemsets",
})
itemset_stats.to_csv(ITEMSET_CSV, index=False)
print(f"Saved: {ITEMSET_CSV}")

COLORS = {"hep": "#e05c5c", "dfhoi": "#5c8de0", "aura_hoi": "#3dba6b"}
LABELS = {"hep": "HEP",     "dfhoi": "DFHOI",   "aura_hoi": "AURA-HOI"}
METRICS = [
    ("time_s",      "Runtime (s)"),
    ("peak_ram_mb", "Peak RAM (MB)"),
]

for dataset in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == dataset].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Dataset: {dataset}", fontsize=14, fontweight="bold", y=1.02)

    for ax, (col, ylabel) in zip(axes, METRICS):
        has_data = False
        for algo in ["hep", "dfhoi", "aura_hoi"]:
            s = sub[sub["algorithm"] == algo].sort_values("alpha").dropna(subset=[col])
            if s.empty or (s[col] == 0).all():
                continue
            ax.plot(
                s["alpha"], s[col],
                marker="o", linewidth=2.0, markersize=6,
                color=COLORS[algo], label=LABELS[algo],
            )
            has_data = True

        ax.set_xlabel("Shared support/occupancy threshold α", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        if has_data:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT / f"{dataset}_comparison.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out.name}")
    plt.close(fig)

print("Charts done.")
PYEOF

echo ""
echo "✓ Charts → docs/core/AURA-HOI/<dataset>_comparison.png"
echo "✓ Itemset counts → ${ITEMSET_CSV}"
