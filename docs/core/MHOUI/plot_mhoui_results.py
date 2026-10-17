#!/usr/bin/env python3
"""Multi-Miner Scalability Plotter for MHOUI.

Produces publication-grade comparative performance profiles (PNG and PDF)
following the exact style of docs/core/MFHOI/plot_mfhoi_results.py:
  - 3 subplots per dataset:
      1. Runtime (s) [log scale] vs Support Threshold
      2. Peak Memory (MB) vs Support Threshold
      3. Discovered Patterns (Card.) [log scale] vs Support Threshold
  - Formatted for top-tier venues (JIIS, Springer, IEEE TKDE).

Run from repository root:
  python docs/core/MHOUI/plot_mhoui_results.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path("docs/core/MHOUI")
CSV_PATH = OUTDIR / "mhoui_multiminer_benchmark_results.csv"

if not CSV_PATH.exists():
    print(f"Error: {CSV_PATH} not found. Run run_mhoui_benchmark.py first.")
    exit(1)

df = pd.read_csv(CSV_PATH)

ALGO_CONFIG = {
    "direct_strong_mhoui": {"label": "MHOUI-Miner (Ours, Direct)", "color": "#d62728", "marker": "o", "lw": 2.4, "ms": 6, "zorder": 10},
    "houi_filter":         {"label": "HOUI + Post-Filter",          "color": "#1f77b4", "marker": "s", "lw": 1.6, "ms": 5, "zorder": 5},
    "ocean_filter":        {"label": "OCEAN + Post-Filter",         "color": "#ff7f0e", "marker": "^", "lw": 1.6, "ms": 5, "zorder": 4},
    "huopm_filter":        {"label": "HUOPM + Post-Filter",         "color": "#9467bd", "marker": "v", "lw": 1.6, "ms": 5, "zorder": 4},
    "il_huopm_filter":     {"label": "IL-HUOPM + Post-Filter",      "color": "#17becf", "marker": "d", "lw": 1.6, "ms": 5, "zorder": 3},
}

METRICS = [
    ("time_total_s",  "Runtime (s)",           True),
    ("peak_ram_mb",   "Peak Memory (MB)",      False),
    ("visited_nodes", "Visited Search Nodes",  True),
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10

for dataset in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == dataset].copy()
    occ_val = sub["minocc"].iloc[0]
    util_val = sub["minutil_ratio"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    fig.suptitle(f"Multi-Miner Scalability Profile: {dataset.capitalize()} (minocc={occ_val}, minutil_ratio={util_val})",
                 fontsize=12, fontweight="bold", y=1.03)

    for i, (ax, (metric_col, ylabel, is_log)) in enumerate(zip(axes, METRICS)):
        for algo, cfg in ALGO_CONFIG.items():
            s = sub[sub["algorithm"] == algo].sort_values("minsup")
            if s.empty:
                continue
            ax.plot(
                s["minsup"], s[metric_col],
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=cfg["lw"],
                markersize=cfg["ms"],
                zorder=cfg["zorder"]
            )

        ax.set_xlabel("Minimum Support (minsup)", fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        if is_log:
            valid_vals = sub[sub[metric_col] > 0][metric_col]
            if not valid_vals.empty:
                min_v = valid_vals.min() * 0.5
                max_v = valid_vals.max() * 2.0
                ax.set_ylim(bottom=max(1e-4, min_v), top=max_v)
                ax.set_yscale("log")

        if i == 1:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_file = OUTDIR / f"{dataset}_multiminer_comparison.{ext}"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        print(f"Generated: {out_file.name}")
    plt.close(fig)

print("All 5 comparative multi-miner charts created successfully.")
