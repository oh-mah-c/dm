#!/usr/bin/env python3
import csv
import argparse
import math
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "closed_hoi_compare"
LOGS = OUT / "logs"
CHARTS = OUT / "charts"
DM = ROOT / "bin" / "dm.exe"

DATASETS = {
    "mushrooms": ROOT / "datasets" / "itemsets" / "mushrooms.txt",
    "foodmart": ROOT / "datasets" / "itemsets" / "foodmartFIM.txt",
}

THRESHOLDS = {
    "mushrooms": [0.10, 0.20, 0.30, 0.40],
    "foodmart": [0.10, 0.20, 0.30],
}

ALGORITHMS = ["cloe_hoi", "hep", "dfhoi", "nam_hep"]

ALGO_LABELS = {
    "cloe_hoi": "CLOE-HOI",
    "hep": "HEP",
    "dfhoi": "DFHOI",
    "nam_hep": "NAM-HEP",
}

ALGO_COLORS = {
    "cloe_hoi": "#005AB5",
    "hep": "#DC3220",
    "dfhoi": "#009E73",
    "nam_hep": "#7A3E9D",
}

ALGO_MARKERS = {
    "cloe_hoi": "o",
    "hep": "s",
    "dfhoi": "^",
    "nam_hep": "D",
}

ALGO_HATCHES = {
    "cloe_hoi": "",
    "hep": "///",
    "dfhoi": "\\\\\\",
    "nam_hep": "...",
}

CSV_FIELDS = ["dataset", "transactions", "algorithm", "alpha", "status", "runtime_sec",
              "peak_ram_mb", "result_ram_bytes", "result_disk_bytes", "itemsets",
              "total_items", "visited_nodes", "pruned_support", "pruned_backward",
              "pruned_envelope", "closure_jumps", "log"]


def apply_paper_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "grid.color": "#D6D6D6",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.65,
        "lines.linewidth": 2.15,
        "lines.markersize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_summary(csv_path: Path):
    numeric = {
        "transactions": int,
        "alpha": float,
        "runtime_sec": float,
        "peak_ram_mb": float,
        "result_ram_bytes": int,
        "result_disk_bytes": int,
        "itemsets": int,
        "total_items": int,
        "visited_nodes": int,
        "pruned_support": int,
        "pruned_backward": int,
        "pruned_envelope": int,
        "closure_jumps": int,
    }
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, cast in numeric.items():
                row[key] = cast(float(row[key])) if cast is int else cast(row[key])
            rows.append(row)
    return rows


def bytes_to_mb(value):
    return value / (1024.0 * 1024.0)


def positive_or_floor(value, floor=1e-3):
    return value if value > 0 else floor


def format_decimal(x, _pos=None):
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}"
    return f"{x:.3f}"


def save_figure(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(CHARTS / f"{stem}.{ext}")


def transaction_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for line in f if line.strip())


def safe_name(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def command(algo: str, dataset: Path, alpha: float, ntrans: int):
    if algo == "cloe_hoi":
        return [str(DM), "cloe_hoi", str(dataset), "0", f"{alpha:.8f}"]
    if algo == "hep":
        return [str(DM), "hep", str(dataset), "0", f"{alpha:.8f}"]
    if algo == "dfhoi":
        return [str(DM), "dfhoi", str(dataset), "0", f"{alpha:.8f}"]
    if algo == "nam_hep":
        minsup = math.ceil(alpha * ntrans)
        return [str(DM), "nam_hep", str(dataset), "0", str(minsup), f"{alpha:.8f}"]
    raise ValueError(algo)


def parse_output(text: str):
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
        "visited_nodes": i(r"visited_nodes=(\d+)"),
        "pruned_support": i(r"pruned_support=(\d+)"),
        "pruned_backward": i(r"pruned_backward=(\d+)"),
        "pruned_envelope": i(r"pruned_envelope=(\d+)"),
        "closure_jumps": i(r"closure_jumps=(\d+)"),
    }


def run_case(dataset_name: str, dataset: Path, alpha: float, algo: str):
    ntrans = transaction_count(dataset)
    cmd = command(algo, dataset, alpha, ntrans)
    log_path = LOGS / f"{algo}_{dataset_name}_a{safe_name(alpha)}.log"
    status = "OK"
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=90)
        text = proc.stdout
        if proc.returncode != 0:
            status = f"EXIT_{proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        status = "TIMEOUT"
    log_path.write_text(text, encoding="utf-8")
    row = {
        "dataset": dataset_name,
        "transactions": ntrans,
        "algorithm": algo,
        "alpha": alpha,
        "status": status,
        "log": str(log_path.relative_to(ROOT)),
    }
    row.update(parse_output(text))
    return row


def plot_lines(rows, metric, ylabel, stem, logy=False, unit_transform=None, note_zero=False):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.8, 4.6), sharey=False)
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, DATASETS):
        subset = [r for r in rows if r["dataset"] == dataset and r["status"] == "OK"]
        for algo in ALGORITHMS:
            vals = sorted([r for r in subset if r["algorithm"] == algo], key=lambda x: x["alpha"])
            if not vals:
                continue
            y = [r[metric] for r in vals]
            if unit_transform:
                y = [unit_transform(v) for v in y]
            if logy:
                y = [positive_or_floor(v) for v in y]
            ax.plot(
                [r["alpha"] for r in vals],
                y,
                marker=ALGO_MARKERS[algo],
                color=ALGO_COLORS[algo],
                label=ALGO_LABELS[algo],
            )
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{dataset} ({subset[0]['transactions']:,} transactions)" if subset else dataset)
        ax.set_xlabel(r"Minimum occupancy threshold $\alpha$")
        ax.grid(True, which="major", axis="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.2f}"))
        if note_zero:
            zero_count = sum(1 for r in subset if r[metric] == 0)
            if zero_count:
                ax.text(
                    0.02,
                    0.05,
                    "zero outputs plotted at floor",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#555555",
                )
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ALGORITHMS), frameon=False,
               bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, stem)
    plt.close(fig)


def plot_itemsets(rows):
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset and r["status"] == "OK"]
        alphas = sorted({r["alpha"] for r in subset})
        width = 0.18
        offsets = [(-1.5 + i) * width for i in range(len(ALGORITHMS))]
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        all_counts = [r["itemsets"] for r in subset]
        use_log = max(all_counts) > 0
        for offset, algo in zip(offsets, ALGORITHMS):
            vals = []
            for alpha in alphas:
                row = next((r for r in subset if r["algorithm"] == algo and r["alpha"] == alpha), None)
                vals.append(row["itemsets"] if row else 0)
            ax.bar(
                [x + offset for x in range(len(alphas))],
                [positive_or_floor(v, 0.5) for v in vals] if use_log else vals,
                width=width,
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                edgecolor="#222222",
                linewidth=0.45,
                hatch=ALGO_HATCHES[algo],
            )
            for i, v in enumerate(vals):
                if v > 0:
                    ax.text(i + offset, positive_or_floor(v, 0.5) * 1.06, f"{v:,}",
                            ha="center", va="bottom", fontsize=7.6, rotation=90)
        if use_log:
            ax.set_yscale("log")
            ax.set_ylabel("Number of itemsets (log scale)")
        else:
            ax.set_ylim(0, 1)
            ax.set_yticks([0])
            ax.set_ylabel("Number of itemsets")
            ax.text(0.5, 0.55, "No itemsets emitted by any method at these thresholds",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#444444")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.2f}" for a in alphas])
        ax.set_xlabel(r"Minimum occupancy threshold $\alpha$")
        ntrans = subset[0]["transactions"] if subset else 0
        ax.set_title(f"Output volume by threshold on {dataset} ({ntrans:,} transactions)")
        ax.grid(True, which="major", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#BBBBBB")
        if use_log:
            ax.text(0.01, 0.03, "zero-count bars plotted at 0.5 floor for log-scale visibility",
                    transform=ax.transAxes, fontsize=8, color="#555555")
        fig.tight_layout()
        save_figure(fig, f"itemsets_bar_{dataset}")
        plt.close(fig)


def plot_efficiency_frontier(rows):
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset and r["status"] == "OK"]
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for algo in ALGORITHMS:
            vals = sorted([r for r in subset if r["algorithm"] == algo], key=lambda x: x["alpha"])
            if not vals:
                continue
            ax.scatter(
                [positive_or_floor(r["runtime_sec"], 1e-5) for r in vals],
                [positive_or_floor(r["peak_ram_mb"], 1e-3) for r in vals],
                s=[44 + math.log10(max(r["itemsets"], 1)) * 24 for r in vals],
                marker=ALGO_MARKERS[algo],
                color=ALGO_COLORS[algo],
                edgecolor="#222222",
                linewidth=0.45,
                alpha=0.92,
                label=ALGO_LABELS[algo],
            )
            for r in vals:
                ax.annotate(
                    f"{r['alpha']:.2f}",
                    (positive_or_floor(r["runtime_sec"], 1e-5), positive_or_floor(r["peak_ram_mb"], 1e-3)),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="#333333",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Runtime (seconds, log scale)")
        ax.set_ylabel("Peak RAM (MB, log scale)")
        ax.set_title(f"Efficiency frontier on {dataset}")
        ax.grid(True, which="major", axis="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="#BBBBBB")
        ax.text(0.01, 0.02, "marker size encodes output itemset count",
                transform=ax.transAxes, fontsize=8, color="#555555")
        fig.tight_layout()
        save_figure(fig, f"efficiency_frontier_{dataset}")
        plt.close(fig)


def write_chart_manifest(rows):
    manifest = CHARTS / "chart_manifest.md"
    lines = [
        "# Closed HOI Q1-Style Benchmark Charts",
        "",
        "All figures are generated from `results/closed_hoi_compare/closed_hoi_summary.csv`.",
        "PNG is for quick viewing; PDF/SVG are vector-ready for LaTeX.",
        "",
        "## Figures",
        "",
        "- `time_line.{png,pdf,svg}`: runtime over occupancy threshold.",
        "- `ram_line.{png,pdf,svg}`: peak RAM over occupancy threshold.",
        "- `disk_line.{png,pdf,svg}`: estimated output disk footprint over occupancy threshold.",
        "- `itemsets_bar_<dataset>.{png,pdf,svg}`: grouped itemset counts by threshold.",
        "- `efficiency_frontier_<dataset>.{png,pdf,svg}`: runtime/RAM frontier with output volume encoded by marker size.",
        "",
        "## Notes",
        "",
        "- Zero-valued output metrics are plotted at a tiny floor only when a logarithmic axis is required.",
        "- CLOE-HOI reports support-closed high-occupancy itemsets; HEP/DFHOI/NAM-HEP use their own repository semantics, so output counts compare behavior and volume rather than exact equivalence.",
        "",
        "## Data points",
        "",
        "| Dataset | Algorithm | alpha | Time (s) | Peak RAM (MB) | Disk (MB) | Itemsets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["status"] != "OK":
            continue
        lines.append(
            f"| {r['dataset']} | {ALGO_LABELS.get(r['algorithm'], r['algorithm'])} | "
            f"{r['alpha']:.2f} | {r['runtime_sec']:.6g} | {r['peak_ram_mb']:.3g} | "
            f"{bytes_to_mb(r['result_disk_bytes']):.3g} | {r['itemsets']:,} |"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_all(rows):
    CHARTS.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    plot_lines(rows, "runtime_sec", "Runtime (seconds)", "time_line", logy=True)
    plot_lines(rows, "peak_ram_mb", "Peak resident memory (MB)", "ram_line", logy=True)
    plot_lines(rows, "result_disk_bytes", "Estimated result size (MB)", "disk_line",
               logy=True, unit_transform=bytes_to_mb, note_zero=True)
    plot_itemsets(rows)
    plot_efficiency_frontier(rows)
    write_chart_manifest(rows)


def main():
    parser = argparse.ArgumentParser(description="Run and plot closed HOI benchmark comparisons.")
    parser.add_argument("--plots-only", action="store_true",
                        help="regenerate publication-quality figures from the existing summary CSV")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    csv_path = OUT / "closed_hoi_summary.csv"
    if args.plots_only:
        plot_all(load_summary(csv_path))
        print(f"[done] publication charts: {CHARTS}")
        return

    subprocess.run(["make", "bin/dm.exe"], cwd=ROOT, check=True)
    rows = []
    for dataset_name, dataset in DATASETS.items():
        for alpha in THRESHOLDS[dataset_name]:
            for algo in ALGORITHMS:
                print(f"[run] dataset={dataset_name} alpha={alpha:.2f} algo={algo}", flush=True)
                rows.append(run_case(dataset_name, dataset, alpha, algo))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    plot_all(rows)
    print(f"[done] {csv_path}")
    print(f"[done] charts: {CHARTS}")


if __name__ == "__main__":
    main()
