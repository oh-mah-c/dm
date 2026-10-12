#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv("docs/core/MHOUI/mhoui_benchmark_results.csv")

os.makedirs("docs/core/MHOUI", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

colors = {
    "houi": "#d95f02",
    "weak_mhoui": "#7570b3",
    "strong_mhoui": "#e7298a",
    "direct_weak_mhoui": "#66a61e",
    "direct_strong_mhoui": "#1b9e77"
}

markers = {
    "houi": "s",
    "weak_mhoui": "^",
    "strong_mhoui": "v",
    "direct_weak_mhoui": "D",
    "direct_strong_mhoui": "o"
}

labels = {
    "houi": "HOUI (Full)",
    "weak_mhoui": "Weak MHOUI",
    "strong_mhoui": "Strong MHOUI",
    "direct_weak_mhoui": "Direct Weak",
    "direct_strong_mhoui": "Direct Strong (Ours)"
}

datasets = ["liquor", "fruithut", "chainstore", "foodmart", "ecommerce"]

# 1. Generate multi-panel charts for datasets
for ds in datasets:
    sub = df[df["dataset"] == ds].copy()
    if sub.empty:
        continue
    
    # We plot sensitivity over different threshold settings (configurations indexed 1..6)
    cfgs = sub[["minsup", "minocc", "minutil_ratio"]].drop_duplicates().reset_index(drop=True)
    cfgs["cfg_id"] = cfgs.index + 1
    cfgs["label"] = cfgs.apply(lambda r: f"C{r.name+1}\n($\\sigma$={r['minsup']}\n$\\beta$={r['minocc']})", axis=1)
    
    merged = pd.merge(sub, cfgs, on=["minsup", "minocc", "minutil_ratio"])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    
    # Panel (a): Runtime
    ax = axes[0]
    for algo in ["houi", "weak_mhoui", "strong_mhoui", "direct_strong_mhoui"]:
        dat = merged[merged["algorithm"] == algo].sort_values("cfg_id")
        ax.plot(dat["cfg_id"], dat["runtime_sec"], marker=markers[algo], color=colors[algo], label=labels[algo], linewidth=1.8, markersize=6)
    ax.set_title(f"(a) Execution Time on {ds.capitalize()}", fontweight="bold")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xticks(cfgs["cfg_id"])
    ax.set_xticklabels(cfgs["label"], fontsize=8)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Panel (b): Peak RAM
    ax = axes[1]
    for algo in ["houi", "weak_mhoui", "strong_mhoui", "direct_strong_mhoui"]:
        dat = merged[merged["algorithm"] == algo].sort_values("cfg_id")
        ax.plot(dat["cfg_id"], dat["peak_ram_mb"], marker=markers[algo], color=colors[algo], label=labels[algo], linewidth=1.8, markersize=6)
    ax.set_title(f"(b) Peak Memory Usage on {ds.capitalize()}", fontweight="bold")
    ax.set_ylabel("Peak RAM (MB)")
    ax.set_xticks(cfgs["cfg_id"])
    ax.set_xticklabels(cfgs["label"], fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Panel (c): Output Itemsets
    ax = axes[2]
    for algo in ["houi", "weak_mhoui", "strong_mhoui", "direct_strong_mhoui"]:
        dat = merged[merged["algorithm"] == algo].sort_values("cfg_id")
        ax.plot(dat["cfg_id"], dat["output_count"], marker=markers[algo], color=colors[algo], label=labels[algo], linewidth=1.8, markersize=6)
    ax.set_title(f"(c) Discovered Patterns on {ds.capitalize()}", fontweight="bold")
    ax.set_ylabel("Number of Patterns")
    ax.set_xticks(cfgs["cfg_id"])
    ax.set_xticklabels(cfgs["label"], fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    out_path = f"docs/core/MHOUI/{ds}_mhoui_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

