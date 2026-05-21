#!/usr/bin/env python3
import csv
import math
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aura_hoi_compare"
LOGS = OUT / "logs"
CHARTS = OUT / "charts"
DM = ROOT / "bin" / "dm.exe"

DATASETS = {
    "mushrooms": ROOT / "datasets" / "itemsets" / "mushrooms.txt",
    "retail": ROOT / "datasets" / "itemsets" / "retail.txt",
}

THRESHOLDS = {
    "mushrooms": [0.10, 0.20, 0.30, 0.40],
    "retail": [0.02, 0.05, 0.10],
}

ALGORITHMS = ["aura_hoi", "hep", "dfhoi"]
LABELS = {"aura_hoi": "AURA-HOI", "hep": "HEP", "dfhoi": "DFHOI"}
COLORS = {"aura_hoi": "#005AB5", "hep": "#DC3220", "dfhoi": "#009E73"}
MARKERS = {"aura_hoi": "o", "hep": "s", "dfhoi": "^"}
HATCHES = {"aura_hoi": "", "hep": "///", "dfhoi": "\\\\\\"}


def transaction_count(path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for line in f if line.strip())


def safe_alpha(alpha):
    return f"{alpha:.2f}".replace(".", "p")


def command(algo, dataset, alpha):
    if algo == "aura_hoi":
        return [str(DM), "aura_hoi", str(dataset), "0", f"{alpha:.8f}", "0", "120", "sum", "raw"]
    return [str(DM), algo, str(dataset), "0", f"{alpha:.8f}"]


def parse_output(text):
    def f(pattern, default=0.0):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    def i(pattern, default=0):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else default

    return {
        "runtime_sec": f(r"Algorithm Core\s+:\s+([0-9.]+) ms") / 1000.0,
        "peak_ram_mb": f(r"Peak RAM .*:\s+([0-9.]+) MB"),
        "result_ram_bytes": i(r"RAM Occupied\s+:\s+[0-9.]+ MB\s+\((\d+) Bytes\)"),
        "result_disk_bytes": i(r"Est\. Disk \(\.txt\)\s+:\s+[0-9.]+ MB\s+\((\d+) Bytes\)"),
        "itemsets": i(r"Frequent Itemsets:\s+(\d+)"),
        "total_items": i(r"Total Items\s+:\s+(\d+)"),
        "raw_accepts": i(r"raw_accepts=(\d+)"),
        "support_classes": i(r"support_classes=(\d+)"),
        "ledger_duplicates": i(r"ledger_duplicates=(\d+)"),
        "visited_nodes": i(r"visited_nodes=(\d+)"),
        "pruned_support": i(r"pruned_support=(\d+)"),
        "pruned_backward": i(r"pruned_backward=(\d+)"),
        "pruned_envelope": i(r"pruned_envelope=(\d+)"),
        "closure_jumps": i(r"closure_jumps=(\d+)"),
    }


def run_case(dataset_name, dataset, alpha, algo):
    log_path = LOGS / f"{algo}_{dataset_name}_a{safe_alpha(alpha)}.log"
    status = "OK"
    try:
        proc = subprocess.run(command(algo, dataset, alpha), cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        text = proc.stdout
        if proc.returncode != 0:
            status = f"EXIT_{proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout if isinstance(exc.stdout, str) else ""
        status = "TIMEOUT"
    log_path.write_text(text, encoding="utf-8")
    row = {
        "dataset": dataset_name,
        "transactions": transaction_count(dataset),
        "algorithm": algo,
        "alpha": alpha,
        "status": status,
        "log": str(log_path.relative_to(ROOT)),
    }
    row.update(parse_output(text))
    return row


def style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 450,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "grid.color": "#D6D6D6",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(CHARTS / f"{stem}.{ext}", bbox_inches="tight")


def floor(v, eps):
    return v if v > 0 else eps


def bytes_to_mb(v):
    return v / (1024.0 * 1024.0)


def plot_line(rows, metric, ylabel, stem, transform=None):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.6, 4.5))
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, DATASETS):
        subset = [r for r in rows if r["dataset"] == dataset and r["status"] == "OK"]
        for algo in ALGORITHMS:
            vals = sorted([r for r in subset if r["algorithm"] == algo], key=lambda x: x["alpha"])
            y = [transform(r[metric]) if transform else r[metric] for r in vals]
            y = [floor(v, 1e-6) for v in y]
            ax.plot([r["alpha"] for r in vals], y, marker=MARKERS[algo], color=COLORS[algo],
                    linewidth=2.2, markersize=6.5, label=LABELS[algo])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        title = f"{dataset} ({subset[0]['transactions']:,} transactions)" if subset else dataset
        ax.set_title(title)
        ax.set_xlabel(r"Minimum occupancy threshold $\alpha$")
        ax.grid(True, which="major", axis="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, stem)
    plt.close(fig)


def plot_itemsets(rows):
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset and r["status"] == "OK"]
        alphas = sorted({r["alpha"] for r in subset})
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        width = 0.23
        offsets = [(-1 + i) * width for i in range(len(ALGORITHMS))]
        max_count = max((r["itemsets"] for r in subset), default=0)
        for offset, algo in zip(offsets, ALGORITHMS):
            vals = []
            for alpha in alphas:
                row = next((r for r in subset if r["algorithm"] == algo and r["alpha"] == alpha), None)
                vals.append(row["itemsets"] if row else 0)
            ax.bar([i + offset for i in range(len(alphas))],
                   [floor(v, 0.5) if max_count else v for v in vals],
                   width=width, label=LABELS[algo], color=COLORS[algo],
                   edgecolor="#222222", linewidth=0.45, hatch=HATCHES[algo])
            for i, v in enumerate(vals):
                if v:
                    ax.text(i + offset, floor(v, 0.5) * 1.06, f"{v:,}",
                            ha="center", va="bottom", fontsize=7.5, rotation=90)
        if max_count:
            ax.set_yscale("log")
            ax.set_ylabel("Number of emitted itemsets (log scale)")
        else:
            ax.set_ylim(0, 1)
            ax.set_ylabel("Number of emitted itemsets")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.2f}" for a in alphas])
        ax.set_xlabel(r"Minimum occupancy threshold $\alpha$")
        ax.set_title(f"HOI output volume on {dataset}")
        ax.grid(True, which="major", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, edgecolor="#BBBBBB")
        ax.text(0.01, 0.03,
                "AURA-HOI emits support-class representatives; HEP/DFHOI emit raw HOI counts.",
                transform=ax.transAxes, fontsize=8, color="#555555")
        fig.tight_layout()
        save(fig, f"itemsets_bar_{dataset}")
        plt.close(fig)


def write_manifest(rows):
    lines = [
        "# AURA-HOI Comparison Charts",
        "",
        "Comparison is intentionally limited to HEP and DFHOI as requested.",
        "AURA-HOI is run in summed-occupancy compatibility mode to match the repository HEP/DFHOI threshold semantics.",
        "AURA-HOI is run in raw-fullset mode; HEP and DFHOI also report raw HOI output.",
        "",
        "Figures:",
        "- `time_line.{png,pdf,svg}`",
        "- `ram_line.{png,pdf,svg}`",
        "- `disk_line.{png,pdf,svg}`",
        "- `itemsets_bar_<dataset>.{png,pdf,svg}`",
        "",
        "| Dataset | Algorithm | alpha | Time (s) | RAM (MB) | Disk (MB) | Itemsets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["status"] == "OK":
            lines.append(f"| {r['dataset']} | {LABELS[r['algorithm']]} | {r['alpha']:.2f} | "
                         f"{r['runtime_sec']:.6g} | {r['peak_ram_mb']:.3g} | "
                         f"{bytes_to_mb(r['result_disk_bytes']):.3g} | {r['itemsets']:,} |")
    (CHARTS / "chart_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["make", "bin/dm.exe"], cwd=ROOT, check=True)

    rows = []
    for dataset_name, dataset in DATASETS.items():
        for alpha in THRESHOLDS[dataset_name]:
            for algo in ALGORITHMS:
                print(f"[run] dataset={dataset_name} alpha={alpha:.2f} algo={algo}", flush=True)
                rows.append(run_case(dataset_name, dataset, alpha, algo))

    fields = ["dataset", "transactions", "algorithm", "alpha", "status", "runtime_sec",
              "peak_ram_mb", "result_ram_bytes", "result_disk_bytes", "itemsets",
              "total_items", "raw_accepts", "support_classes", "ledger_duplicates",
              "visited_nodes", "pruned_support", "pruned_backward", "pruned_envelope",
              "closure_jumps", "log"]
    csv_path = OUT / "aura_hoi_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    style()
    plot_line(rows, "runtime_sec", "Runtime (seconds, log scale)", "time_line")
    plot_line(rows, "peak_ram_mb", "Peak RAM (MB, log scale)", "ram_line")
    plot_line(rows, "result_disk_bytes", "Estimated output disk (MB, log scale)", "disk_line", bytes_to_mb)
    plot_itemsets(rows)
    write_manifest(rows)
    print(f"[done] {csv_path}")
    print(f"[done] charts: {CHARTS}")


if __name__ == "__main__":
    main()
