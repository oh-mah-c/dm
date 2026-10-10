from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV  = Path("docs/core/AURA-HOI/benchmark_results.csv")
OUT  = Path("docs/core/AURA-HOI")
ITEMSET_CSV = OUT / "itemset_counts.csv"

df = pd.read_csv(CSV)
df["alpha"]         = pd.to_numeric(df["alpha"],         errors="coerce")
df["minsup"]        = pd.to_numeric(df["minsup"],        errors="coerce")
df["time_s"]        = pd.to_numeric(df["time_s"],        errors="coerce")
df["peak_ram_mb"]   = pd.to_numeric(df["peak_ram_mb"],   errors="coerce")
df["itemsets"]      = pd.to_numeric(df["itemsets"],      errors="coerce")
df["visited_nodes"] = pd.to_numeric(df["visited_nodes"], errors="coerce")

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
    ("time_s",        "Runtime (s)"),
    ("visited_nodes", "Search Space (Visited Nodes)"),
    ("peak_ram_mb",   "Peak RAM (MB)"),
]

for dataset in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == dataset].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle(f"Dataset: {dataset}", fontsize=13, fontweight="bold", y=1.02)

    for ax, (col, ylabel) in zip(axes, METRICS):
        has_data = False
        for algo in ["hep", "dfhoi", "aura_hoi"]:
            s = sub[sub["algorithm"] == algo].sort_values("alpha").dropna(subset=[col])
            if s.empty:
                continue
            ax.plot(
                s["alpha"], s[col],
                marker="o", linewidth=2.0, markersize=5,
                color=COLORS[algo], label=LABELS[algo],
            )
            has_data = True

        ax.set_xlabel("Shared threshold α", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        if col in ["time_s", "visited_nodes"]:
            valid_vals = sub[sub[col] > 0][col].dropna()
            if not valid_vals.empty:
                ax.set_yscale("log")
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

print("All charts updated successfully.")
