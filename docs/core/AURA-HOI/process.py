"""
AURA-HOI Benchmark: parse logs → CSV → comparison charts.

Each chart = 1 dataset, 3 subplots (Time / RAM / Disk), 3 lines (hep / dfhoi / aura_hoi).
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("docs/core/AURA-HOI/results")

ALGO_COLORS = {
    "hep":      "#e05c5c",
    "dfhoi":    "#5c8de0",
    "aura_hoi": "#3dba6b",
}
ALGO_LABELS = {
    "hep":      "HEP",
    "dfhoi":    "DFHOI",
    "aura_hoi": "AURA-HOI",
}

# ── Regex patterns ────────────────────────────────────────────────────────────
RE_WALL_INTERNAL = re.compile(r"TOTAL WALL TIME\s*:\s*([0-9.]+)\s*ms")
RE_WALL_TIME_EXT = re.compile(r"Elapsed \(wall clock\) time.*?([0-9:]+\.[0-9]+)")
RE_RAM_KB        = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
RE_DISK_MB       = re.compile(r"Est\. Disk \(\.txt\)\s*:\s*([0-9.]+)\s*MB")
RE_ITEMSETS      = re.compile(r"Frequent Itemsets\s*:\s*(\d+)")
RE_ITEMSETS_ALT  = re.compile(r"(?:high occupancy )?itemsets found:\s*(\d+)", re.IGNORECASE)


def parse_wall_ext(s: str) -> float:
    """Parse m:ss.ff or h:mm:ss.ff from /usr/bin/time -v output."""
    parts = s.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(s)


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")

    # Time (prefer internal high-res timer)
    time_s = None
    m = RE_WALL_INTERNAL.search(text)
    if m:
        time_s = float(m.group(1)) / 1000.0
    else:
        m = RE_WALL_TIME_EXT.search(text)
        if m:
            time_s = parse_wall_ext(m.group(1))

    # RAM
    ram_mb = None
    m = RE_RAM_KB.search(text)
    if m:
        ram_mb = int(m.group(1)) / 1024.0

    # Disk
    disk_mb = 0.0
    m = RE_DISK_MB.search(text)
    if m:
        disk_mb = float(m.group(1))

    # Itemset count
    itemsets = 0
    m = RE_ITEMSETS.search(text)
    if m:
        itemsets = int(m.group(1))
    else:
        m = RE_ITEMSETS_ALT.search(text)
        if m:
            itemsets = int(m.group(1))

    return {
        "time_s":    time_s,
        "peak_ram_mb": ram_mb,
        "disk_mb":   disk_mb,
        "itemsets":  itemsets,
        "status":    "OK" if time_s is not None else "FAIL",
    }


# ── Collect rows ──────────────────────────────────────────────────────────────
rows = []

for log_path in sorted(ROOT.glob("raw_logs/*/*/alpha_*/*.txt")):
    parts = log_path.parts
    dataset   = parts[-4]
    algorithm = parts[-3]
    alpha_str = parts[-2].replace("alpha_", "")
    run       = log_path.stem.replace("run_", "")
    if run == "warmup":
        continue

    rec = parse_log(log_path)
    rec.update({
        "dataset":   dataset,
        "algorithm": algorithm,
        "alpha":     float(alpha_str),
        "run":       run,
    })
    rows.append(rec)

if not rows:
    print("No log files found — run benchmark.sh first.", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)

# ── Median summary ────────────────────────────────────────────────────────────
summary = (
    df.groupby(["dataset", "algorithm", "alpha"], as_index=False)
      .agg(
          time_s_median      = ("time_s",       "median"),
          peak_ram_mb_median = ("peak_ram_mb",  "median"),
          disk_mb_median     = ("disk_mb",       "median"),
          itemsets_median    = ("itemsets",      "median"),
      )
)

(ROOT / "summary").mkdir(parents=True, exist_ok=True)
summary.to_csv(ROOT / "summary" / "aura_hoi_benchmark_median.csv", index=False)
df.to_csv(ROOT / "summary" / "aura_hoi_benchmark_runs.csv", index=False)
print("Saved CSVs.")

# ── Charts: 1 figure per dataset, 3 subplots (Time / RAM / Disk) ─────────────
METRICS = [
    ("time_s_median",       "Runtime (s)",         True),
    ("peak_ram_mb_median",  "Peak RAM (MB)",        True),
    ("disk_mb_median",      "Disk footprint (MB)",  False),
]

plot_dir = ROOT / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

for dataset in sorted(summary["dataset"].unique()):
    sub = summary[summary["dataset"] == dataset].sort_values("alpha")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Dataset: {dataset}", fontsize=14, fontweight="bold")

    for ax, (col, ylabel, use_log) in zip(axes, METRICS):
        for algo in ["hep", "dfhoi", "aura_hoi"]:
            s = sub[sub["algorithm"] == algo].dropna(subset=[col])
            if s.empty:
                continue
            ax.plot(
                s["alpha"], s[col],
                marker="o", linewidth=2, markersize=5,
                color=ALGO_COLORS[algo],
                label=ALGO_LABELS[algo],
            )

        ax.set_xlabel("Min-occupancy threshold α", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8)

        if use_log:
            # Only apply log scale if there are positive values
            yvals = sub[[col]].dropna()
            if (yvals > 0).any().any():
                ax.set_yscale("log")

    fig.tight_layout()
    out_pdf = plot_dir / f"{dataset}_comparison.pdf"
    out_png = plot_dir / f"{dataset}_comparison.png"
    fig.savefig(out_pdf, dpi=150)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")

print("Done.")
