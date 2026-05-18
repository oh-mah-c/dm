#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    row = {
        "log": path.name,
        "algorithm": "",
        "dataset": "",
        "setting": "",
        "runner_status": "UNKNOWN",
        "runtime_sec": "",
        "total_sec": "",
        "peak_ram_mb": "",
        "output_count": "",
        "total_output_items": "",
        "result_ram_bytes": "",
        "result_disk_est_bytes": "",
        "visited_nodes": "",
        "pruned_support": "",
        "pruned_utility": "",
        "pruned_occupancy": "",
        "closed_or_hybrid": "",
        "algorithm_name": "",
        "raw_houi_count": "",
        "weak_count": "",
        "strong_count": "",
    }
    for line in text.splitlines():
        if line.startswith("label="):
            row["algorithm"] = line.split("=", 1)[1].strip()
        elif line.startswith("dataset="):
            row["dataset"] = line.split("=", 1)[1].strip()
        elif line.startswith("setting="):
            row["setting"] = line.split("=", 1)[1].strip()
        elif line.startswith("runner_status="):
            row["runner_status"] = line.split("=", 1)[1].strip()

    if "CHUO-Miner" in text:
        row["closed_or_hybrid"] = "closed_hybrid"
        for key in [
            "runtime_sec", "total_sec", "peak_ram_mb", "output_count",
            "total_output_items", "result_ram_bytes", "result_disk_est_bytes",
            "visited_nodes", "pruned_support",
        ]:
            m = re.search(rf"^{key}=([0-9.]+)", text, re.MULTILINE)
            if m:
                row[key] = m.group(1)
        m = re.search(r"^pruned_ruu=([0-9.]+)", text, re.MULTILINE)
        if m:
            row["pruned_utility"] = m.group(1)
        m = re.search(r"^pruned_oub=([0-9.]+)", text, re.MULTILINE)
        if m:
            row["pruned_occupancy"] = m.group(1)
        return row

    name = re.search(r"Algorithm\s+:\s+(.+)", text)
    if name:
        row["algorithm_name"] = name.group(1).strip()
    runtime = re.search(r"Algorithm Core\s+:\s+([0-9.]+)\s+ms", text)
    total = re.search(r"TOTAL WALL TIME\s+:\s+([0-9.]+)\s+ms", text)
    ram = re.search(r"Peak RAM \(VmHWM\)\s+:\s+([0-9.]+)\s+MB", text)
    itemsets = re.search(r"Frequent Itemsets:\s+([0-9]+)", text)
    total_items = re.search(r"Total Items\s+:\s+([0-9]+)", text)
    result_ram = re.search(r"RAM Occupied\s+:\s+[0-9.]+\s+MB\s+\(([0-9]+)\s+Bytes\)", text)
    result_disk = re.search(r"Est\. Disk \(\.txt\)\s+:\s+[0-9.]+\s+MB\s+\(([0-9]+)\s+Bytes\)", text)
    if runtime:
        row["runtime_sec"] = str(float(runtime.group(1)) / 1000.0)
    if total:
        row["total_sec"] = str(float(total.group(1)) / 1000.0)
    if ram:
        row["peak_ram_mb"] = ram.group(1)
    if itemsets:
        row["output_count"] = itemsets.group(1)
    if total_items:
        row["total_output_items"] = total_items.group(1)
    if result_ram:
        row["result_ram_bytes"] = result_ram.group(1)
    if result_disk:
        row["result_disk_est_bytes"] = result_disk.group(1)

    if row["algorithm"] == "mhoui":
        row["closed_or_hybrid"] = "hybrid_not_closed"
        m = re.search(r"HOUI=([0-9]+)\s+Weak=([0-9]+)\s+Strong=([0-9]+)", text)
        if m:
            row["raw_houi_count"] = m.group(1)
            row["weak_count"] = m.group(2)
            row["strong_count"] = m.group(3)
        m = re.search(r"visited=([0-9]+)", text)
        if m:
            row["visited_nodes"] = m.group(1)
        m = re.search(r"pruned_support=([0-9]+)", text)
        if m:
            row["pruned_support"] = m.group(1)
        m = re.search(r"pruned_uub=([0-9]+)", text)
        if m:
            row["pruned_utility"] = m.group(1)
        m = re.search(r"pruned_oub1=([0-9]+)", text)
        if m:
            row["pruned_occupancy"] = m.group(1)
    elif row["algorithm"] in {"chui_miner", "efim_closed"}:
        row["closed_or_hybrid"] = "closed_utility"
    else:
        row["closed_or_hybrid"] = "utility"
    return row


def write_csv(rows, out_dir):
    keys = [
        "dataset", "setting", "algorithm", "closed_or_hybrid", "runner_status",
        "runtime_sec", "total_sec", "peak_ram_mb", "output_count",
        "total_output_items", "result_ram_bytes", "result_disk_est_bytes",
        "visited_nodes", "pruned_support", "pruned_utility", "pruned_occupancy",
        "raw_houi_count", "weak_count", "strong_count", "algorithm_name", "log",
    ]
    with (out_dir / "chuo_q1_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row, key):
    if not row:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def plot_metric(rows, out_dir, metric, title, ylabel, log=False):
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        labels = [f"{r['algorithm']}\n{r.get('setting', '')}" for r in ds_rows]
        vals = [fnum(r, metric) for r in ds_rows]
        colors = ["#2f6f7e" if l == "chuo_miner" else "#7a869a" for l in labels]
        plt.figure(figsize=(8.2, 4.4))
        plt.bar(labels, vals, color=colors)
        plt.ylabel(ylabel)
        plt.title(f"{title}: {ds}")
        if log:
            plt.yscale("log")
        plt.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_{metric}_comparison.png", dpi=180)
        plt.close()


def plot_sensitivity(rows, out_dir, metric, title, ylabel, log=False):
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds and r["algorithm"] in {"chuo_miner", "mhoui"}]
        settings = []
        for r in ds_rows:
            if r.get("setting") not in settings:
                settings.append(r.get("setting"))
        if not settings:
            continue
        plt.figure(figsize=(8.6, 4.6))
        for alg, color in [("chuo_miner", "#2f6f7e"), ("mhoui", "#a65f3f")]:
            vals = []
            for s in settings:
                row = next((r for r in ds_rows if r["algorithm"] == alg and r.get("setting") == s), None)
                vals.append(fnum(row, metric) if row else 0.0)
            plt.plot(range(len(settings)), vals, marker="o", linewidth=2, color=color, label=alg)
        plt.xticks(range(len(settings)), settings, rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{title} sensitivity: {ds}")
        if log:
            plt.yscale("log")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_{metric}_sensitivity.png", dpi=180)
        plt.close()


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/chuo_q1_comparison")
    rows = [parse_log(p) for p in sorted((out_dir / "logs").glob("*.log"))]
    rows.sort(key=lambda r: (r["dataset"], r.get("setting", ""), ["chuo_miner", "mhoui", "chui_miner", "efim_closed", "efim"].index(r["algorithm"])))
    write_csv(rows, out_dir)
    if not rows:
        return 0
    plot_metric(rows, out_dir, "runtime_sec", "Runtime", "Seconds", log=True)
    plot_metric(rows, out_dir, "peak_ram_mb", "Peak RAM", "MB")
    plot_metric(rows, out_dir, "output_count", "Output size", "Patterns", log=True)
    plot_metric(rows, out_dir, "result_disk_est_bytes", "Estimated result disk footprint", "Bytes", log=True)
    plot_sensitivity(rows, out_dir, "runtime_sec", "Runtime", "Seconds", log=True)
    plot_sensitivity(rows, out_dir, "output_count", "Output size", "Patterns", log=True)
    plot_sensitivity(rows, out_dir, "peak_ram_mb", "Peak RAM", "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
