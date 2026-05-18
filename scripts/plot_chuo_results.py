#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_KEYS = {
    "transactions", "max_item_id", "minsup_ratio", "minsup_count", "min_utility",
    "min_occupancy", "load_sec", "runtime_sec", "total_sec", "peak_ram_mb",
    "surviving_items", "visited_nodes", "generated_children", "joined_entries",
    "pruned_twu", "pruned_support", "pruned_ruu", "pruned_oub",
    "pruned_backward", "closure_jumps", "closure_hash_hits", "output_count",
    "total_output_items", "avg_output_length", "avg_support", "avg_utility",
    "avg_occupancy", "best_utility", "best_occupancy", "max_depth",
    "signatures_built", "signature_value_records", "reconstruction_checks",
    "reconstruction_failures", "result_ram_bytes", "result_disk_est_bytes",
}


def parse_file(path: Path):
    rows = []
    current = {}
    config = ""
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            if current:
                current["source_file"] = path.name
                current["config"] = config
                rows.append(current)
                current = {}
            continue
        if line.startswith("====="):
            config = line.strip("= ")
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
        current["config"] = config
        rows.append(current)
    return rows


def dataset_name(row):
    return Path(str(row.get("input", row.get("source_file", "dataset")))).stem


def setting_label(row):
    return f"s={row.get('minsup_ratio', 0):.3g}\no={row.get('min_occupancy', 0):.2g}"


def write_csv(rows, out_dir):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with (out_dir / "chuo_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, out_dir, metric, title, ylabel):
    grouped = {}
    for row in rows:
        grouped.setdefault(dataset_name(row), []).append(row)
    for ds, ds_rows in grouped.items():
        xs = range(len(ds_rows))
        ys = [float(r.get(metric, 0.0)) for r in ds_rows]
        plt.figure(figsize=(7.2, 4.2))
        bars = plt.bar(xs, ys, color="#476f8f")
        plt.xticks(list(xs), [setting_label(r) for r in ds_rows])
        plt.ylabel(ylabel)
        plt.title(f"{title}: {ds}")
        plt.grid(axis="y", alpha=0.25)
        for b, y in zip(bars, ys):
            plt.text(b.get_x() + b.get_width() / 2, y, f"{y:.3g}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_{metric}.png", dpi=180)
        plt.close()


def plot_pruning(rows, out_dir):
    for row in rows:
        ds = dataset_name(row)
        suffix = f"s{row.get('minsup_ratio', 0):.3g}_o{row.get('min_occupancy', 0):.2g}".replace(".", "p")
        labels = ["TWU", "support", "RUU", "OUB", "backward"]
        vals = [row.get("pruned_twu", 0), row.get("pruned_support", 0), row.get("pruned_ruu", 0),
                row.get("pruned_oub", 0), row.get("pruned_backward", 0)]
        plt.figure(figsize=(6.4, 3.8))
        plt.bar(labels, vals, color=["#667085", "#2f6f7e", "#a65f3f", "#4d875c", "#7b5ea7"])
        plt.ylabel("Pruned nodes/items")
        plt.title(f"CHUO pruning profile: {ds}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_pruning_{suffix}.png", dpi=180)
        plt.close()


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/chuo_compare_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(out_dir.glob("chuo_*.txt")):
        rows.extend(parse_file(path))
    write_csv(rows, out_dir)
    if not rows:
        return 0
    plot_metric(rows, out_dir, "runtime_sec", "Runtime", "Seconds")
    plot_metric(rows, out_dir, "peak_ram_mb", "Peak RAM", "MB")
    plot_metric(rows, out_dir, "output_count", "Closed HUO itemsets", "CHUOIs")
    plot_metric(rows, out_dir, "avg_utility", "Average utility", "Utility")
    plot_metric(rows, out_dir, "avg_occupancy", "Average occupancy", "Occupancy")
    plot_metric(rows, out_dir, "closure_jumps", "Closure jumps", "Jumps")
    plot_pruning(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
