#!/usr/bin/env python3
"""Plotting script for MHOUI-Miner Ablation Study.

Generates publication-quality figures (PNG and PDF) aligned with JIIS/Springer/IEEE standards:
  1. Runtime speedup across pruning ablations (log-scale execution time).
  2. Memory footprint comparison (Direct Single-Phase vs Two-Stage Materialization).
  3. Search-space reduction (Candidates pruned by Joint Bounds and Branch Dominance).

Run from repository root:
  python docs/core/MHOUI/plot_ablation_results.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "docs/core/MHOUI/ablation_benchmark_results.csv"
OUT_DIR = "docs/core/MHOUI"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

# Consistent scientific color palette
COLORS = {
    "direct_strong_mhoui": "#1b9e77",  # Deep Teal / Green (Ours)
    "ablation_no_dom":     "#377eb8",  # Steel Blue
    "ablation_no_oub":     "#984ea3",  # Purple
    "ablation_no_uub":     "#ff7f00",  # Amber / Orange
    "ablation_no_bounds":  "#e41a1c",  # Crimson
    "houi_filter":         "#999999"   # Neutral Gray (Two-stage Baseline)
}

LABELS = {
    "direct_strong_mhoui": "Full MHOUI-Miner (Ours)",
    "ablation_no_dom":     "w/o Branch Dominance",
    "ablation_no_oub":     "w/o Occupancy Envelope",
    "ablation_no_uub":     "w/o Utility Bound",
    "ablation_no_bounds":  "w/o Joint Bounds",
    "houi_filter":         "Two-Stage Baseline"
}


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run run_ablation_study.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Multi-dataset Ablation Runtime & Memory Comparison
    datasets = df["dataset"].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # Average across configurations per dataset
    avg_df = df.groupby(["dataset", "algorithm"]).agg({
        "time_total_s": "mean",
        "peak_ram_mb": "mean",
        "candidates": "mean",
        "maximal": "mean"
    }).reset_index()

    algo_order = [
        "direct_strong_mhoui",
        "ablation_no_dom",
        "ablation_no_oub",
        "ablation_no_uub",
        "ablation_no_bounds",
        "houi_filter"
    ]
    # Filter only available algorithms
    algo_order = [a for a in algo_order if a in avg_df["algorithm"].unique()]

    # Panel (a): Normalized Runtime Relative to Full MHOUI-Miner
    ax = axes[0]
    bar_width = 0.13
    x = np.arange(n_datasets)

    for i, algo in enumerate(algo_order):
        vals = []
        for ds in datasets:
            sub = avg_df[(avg_df["dataset"] == ds) & (avg_df["algorithm"] == algo)]
            vals.append(sub["time_total_s"].values[0] if not sub.empty else 0.0)
        
        offset = (i - len(algo_order) / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, "#333333"), edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.set_title("(a) Execution Time across Algorithmic Ablations", fontweight="bold", pad=10)
    ax.set_ylabel("Execution Time (seconds) [log-scale]", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in datasets], fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, axis="y")
    ax.legend(framealpha=0.95, fontsize=8, loc="upper right")

    # Panel (b): Peak RAM Usage
    ax = axes[1]
    for i, algo in enumerate(algo_order):
        vals = []
        for ds in datasets:
            sub = avg_df[(avg_df["dataset"] == ds) & (avg_df["algorithm"] == algo)]
            vals.append(sub["peak_ram_mb"].values[0] if not sub.empty else 0.0)
        
        offset = (i - len(algo_order) / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, "#333333"), edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.set_title("(b) Peak Memory Consumption across Algorithmic Ablations", fontweight="bold", pad=10)
    ax.set_ylabel("Peak RAM (MB)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in datasets], fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, axis="y")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_p = os.path.join(OUT_DIR, f"mhoui_ablation_profile.{ext}")
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_p}")
    plt.close()

    print("Ablation charts generated successfully.")


if __name__ == "__main__":
    main()
