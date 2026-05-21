#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_KEYS = {
    "transactions", "semantic_concepts", "total_prompt_tokens", "avg_prompt_tokens",
    "minsup_ratio", "minsup_count", "theta", "alpha_min", "runtime_sec",
    "total_sec", "peak_ram_mb", "candidate_optimizers", "pareto_removed",
    "frequent_singletons", "singleton_occurrences", "visited_nodes", "joins",
    "joined_entries", "pruned_support", "pruned_ptwo", "pruned_aaub",
    "output_count", "total_output_items", "avg_output_length", "avg_support",
    "avg_utility", "avg_alignment", "best_utility", "best_alignment",
    "max_depth", "result_ram_bytes", "result_disk_est_bytes",
}


def parse_file(path: Path):
    rows = []
    current = {}
    cfg = ""
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            if current:
                current["source_file"] = path.name
                current["config"] = cfg
                rows.append(current)
                current = {}
            continue
        if line.startswith("====="):
            cfg = line.strip("= ")
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key in NUMERIC_KEYS:
            try:
                current[key] = float(value)
            except ValueError:
                current[key] = value
        else:
            current[key] = value
    if current:
        current["source_file"] = path.name
        current["config"] = cfg
        rows.append(current)
    return rows


def save_csv(rows, out_dir: Path):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with (out_dir / "hupp_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def label(row):
    return f"s={row.get('minsup_ratio', 0):.3g}\na={row.get('alpha_min', 0):.2g}"


def plot_metric(rows, out_dir: Path, metric: str, title: str, ylabel: str):
    by_dataset = {}
    for row in rows:
        ds = Path(str(row.get("input", row.get("source_file", "dataset")))).stem
        by_dataset.setdefault(ds, []).append(row)
    for ds, ds_rows in by_dataset.items():
        xs = list(range(len(ds_rows)))
        ys = [float(r.get(metric, 0.0)) for r in ds_rows]
        plt.figure(figsize=(7.2, 4.2))
        bars = plt.bar(xs, ys, color="#2f6f7e")
        plt.xticks(xs, [label(r) for r in ds_rows])
        plt.ylabel(ylabel)
        plt.title(f"{title}: {ds}")
        plt.grid(axis="y", alpha=0.25)
        for b, y in zip(bars, ys):
            plt.text(b.get_x() + b.get_width() / 2, y, f"{y:.3g}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_{metric}.png", dpi=180)
        plt.close()


def plot_pruning(rows, out_dir: Path):
    for row in rows:
        ds = Path(str(row.get("input", row.get("source_file", "dataset")))).stem
        suffix = f"s{row.get('minsup_ratio', 0):.3g}_a{row.get('alpha_min', 0):.2g}".replace(".", "p")
        vals = [row.get("pruned_support", 0), row.get("pruned_ptwo", 0), row.get("pruned_aaub", 0)]
        plt.figure(figsize=(5.8, 3.8))
        plt.bar(["support", "PTWO", "AAUB"], vals, color=["#566b8f", "#a65f3f", "#4d875c"])
        plt.ylabel("Pruned nodes")
        plt.title(f"HUPP pruning profile: {ds}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_pruning_{suffix}.png", dpi=180)
        plt.close()


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/hupp_compare_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(out_dir.glob("hupp_*.txt")):
        rows.extend(parse_file(path))
    save_csv(rows, out_dir)
    if not rows:
        return 0
    plot_metric(rows, out_dir, "runtime_sec", "Runtime", "Seconds")
    plot_metric(rows, out_dir, "output_count", "High-utility prompt patterns", "Patterns")
    plot_metric(rows, out_dir, "peak_ram_mb", "Peak RAM", "MB")
    plot_metric(rows, out_dir, "avg_utility", "Average utility", "Utility")
    plot_metric(rows, out_dir, "avg_alignment", "Average alignment", "Alignment")
    plot_pruning(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
