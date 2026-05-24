#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# AURA-HOI Benchmark — runs all algorithms, collects metrics, plots charts
# Usage: ./docs/core/AURA-HOI/benchmark.sh
# Outputs: docs/core/AURA-HOI/benchmark_results.csv
#          docs/core/AURA-HOI/<dataset>_comparison.png  (4 charts)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BIN="./bin/dm.exe"
MAXSEC=300
OUTDIR="docs/core/AURA-HOI"
CSV="${OUTDIR}/benchmark_results.csv"
TMP="$(mktemp)"

# ── Init CSV ────────────────────────────────────────────────────────────
echo "dataset,algorithm,alpha,minsup,time_s,peak_ram_mb,disk_mb,itemsets" > "$CSV"

# ── run_one: execute one (dataset, algo, alpha) and append a CSV row ────
run_one() {
  local ds="$1" algo="$2" alpha="$3" minsup="$4"
  local dspath="datasets/itemsets/${ds}.txt"
  echo "  → ${ds} | ${algo} | α=${alpha}"

  if [ "$algo" = "aura_hoi" ]; then
    /usr/bin/time -v "$BIN" aura_hoi "$dspath" 0 "$alpha" "$minsup" "$MAXSEC" sum raw > "$TMP" 2>&1 || true
  else
    /usr/bin/time -v "$BIN" "$algo" "$dspath" 0 "$alpha" "$minsup" "$MAXSEC" > "$TMP" 2>&1 || true
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

  disk_mb=$(grep -oP "Est\. Disk \(\.txt\)\s*:\s*\K[0-9.]+" "$TMP" 2>/dev/null | head -1 || echo "0")

  itemsets=$(grep -oP "Frequent Itemsets\s*:\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "")
  if [ -z "$itemsets" ]; then
    itemsets=$(grep -oiP "(?:high occupancy )?itemsets found:\s*\K\d+" "$TMP" 2>/dev/null | head -1 || echo "0")
  fi

  echo "${ds},${algo},${alpha},${minsup},${time_s:-},${ram_mb},${disk_mb:-0},${itemsets:-0}" >> "$CSV"
}

# ── Threshold matrix ────────────────────────────────────────────────────
declare -A MINSUP=([mushrooms]=85 [chess]=32 [retail]=89 [T10I4D100K]=100)
declare -A ALPHAS=(
  [mushrooms]="0.05 0.075 0.10 0.125 0.15 0.20 0.30 0.40"
  [chess]="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.70"
  [retail]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
  [T10I4D100K]="0.005 0.01 0.02 0.03 0.05 0.075 0.10"
)

ALGOS="hep dfhoi aura_hoi"

# ── Run all configurations ──────────────────────────────────────────────
for ds in mushrooms chess retail T10I4D100K; do
  echo "▶ Dataset: $ds (minsup=${MINSUP[$ds]})"
  for alpha in ${ALPHAS[$ds]}; do
    for algo in $ALGOS; do
      run_one "$ds" "$algo" "$alpha" "${MINSUP[$ds]}"
    done
  done
done

rm -f "$TMP"
echo ""
echo "✓ Results saved → ${CSV}"

# ── Generate charts with inline Python ─────────────────────────────────
VENV_PY="./.venv/bin/python3"
[ -x "$VENV_PY" ] || VENV_PY="python3"

$VENV_PY - <<'PYEOF'
import sys, re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CSV  = Path("docs/core/AURA-HOI/benchmark_results.csv")
OUT  = Path("docs/core/AURA-HOI")

df = pd.read_csv(CSV)
df["time_s"]      = pd.to_numeric(df["time_s"],      errors="coerce")
df["peak_ram_mb"] = pd.to_numeric(df["peak_ram_mb"], errors="coerce")
df["disk_mb"]     = pd.to_numeric(df["disk_mb"],     errors="coerce")
df["itemsets"]    = pd.to_numeric(df["itemsets"],     errors="coerce")

COLORS = {"hep": "#e05c5c", "dfhoi": "#5c8de0", "aura_hoi": "#3dba6b"}
LABELS = {"hep": "HEP",     "dfhoi": "DFHOI",   "aura_hoi": "AURA-HOI"}
METRICS = [
    ("time_s",      "Runtime (s)"),
    ("peak_ram_mb", "Peak RAM (MB)"),
    ("disk_mb",     "Disk (MB)"),
]

for dataset in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == dataset].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
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

        ax.set_xlabel("Min-occupancy α", fontsize=9)
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
