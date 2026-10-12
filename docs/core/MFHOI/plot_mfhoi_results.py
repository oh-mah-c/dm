from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path("docs/core/MFHOI")
CSV_PATH = OUTDIR / "mfhoi_benchmark_results.csv"

df = pd.read_csv(CSV_PATH)

ALGO_CONFIG = {
    "mfhoi":    {"label": "MFHOI-Miner (Ours)", "color": "#d62728", "marker": "o", "lw": 2.4, "ms": 6, "zorder": 10},
    "genmax":   {"label": "GenMax (MFI)",        "color": "#1f77b4", "marker": "s", "lw": 1.6, "ms": 5, "zorder": 5},
    "mafia":    {"label": "MAFIA (MFI)",         "color": "#ff7f0e", "marker": "^", "lw": 1.6, "ms": 5, "zorder": 5},
    "lcm":      {"label": "LCM (Closed FCI)",    "color": "#9467bd", "marker": "v", "lw": 1.6, "ms": 5, "zorder": 4},
    "hep":      {"label": "HEP (Raw HOI)",       "color": "#8c564b", "marker": "d", "lw": 1.6, "ms": 5, "zorder": 3},
    "dfhoi":    {"label": "DFHOI (Raw HOI)",     "color": "#e377c2", "marker": "x", "lw": 1.6, "ms": 5, "zorder": 3},
}

METRICS = [
    ("time_s",      "Runtime (s)",                     True),
    ("peak_ram_mb", "Peak Memory (MB)",                False),
    ("patterns",    "Discovered Patterns (Card.)",     True),
]

for dataset in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == dataset].copy()
    beta_val = sub["beta"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    fig.suptitle(f"Multi-Miner Scalability Profile: {dataset} (β={beta_val})", fontsize=12, fontweight="bold", y=1.03)

    for i, (ax, (metric_col, ylabel, is_log)) in enumerate(zip(axes, METRICS)):
        for algo, cfg in ALGO_CONFIG.items():
            s = sub[sub["algorithm"] == algo].sort_values("alpha")
            if s.empty:
                continue
            ax.plot(
                s["alpha"], s[metric_col],
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=cfg["lw"],
                markersize=cfg["ms"],
                zorder=cfg["zorder"]
            )

        ax.set_xlabel("Support Threshold (α)", fontsize=9, fontweight="bold")
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
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        print(f"Generated: {out_file.name}")
    plt.close(fig)

print("All 5 comparative multi-miner charts created successfully.")
