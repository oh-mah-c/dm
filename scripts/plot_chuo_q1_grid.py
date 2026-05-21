#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGORITHMS = ["chuo_miner", "mhoui", "chui_miner", "efim_closed", "efim"]


def parse_block(block, source):
    header = block[0] if block else ""
    row = {"source_file": source.name, "runner_status": "UNKNOWN"}
    m = re.search(r"algorithm=(\S+)\s+dataset=(\S+)\s+minutil=([0-9.]+)\s+minsup=([0-9.]+)\s+minocc=([0-9.]+)", header)
    if not m:
        return None
    row.update({
        "algorithm": m.group(1),
        "dataset": m.group(2),
        "minutil": float(m.group(3)),
        "minsup": float(m.group(4)),
        "minocc": float(m.group(5)),
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
        "raw_houi_count": "",
        "weak_count": "",
        "strong_count": "",
    })
    text = "\n".join(block)
    status = re.search(r"runner_status=(\S+)", text)
    if status:
        row["runner_status"] = status.group(1)

    if row["algorithm"] == "chuo_miner":
        for key in ["runtime_sec", "total_sec", "peak_ram_mb", "output_count", "total_output_items",
                    "result_ram_bytes", "result_disk_est_bytes", "visited_nodes", "pruned_support"]:
            mm = re.search(rf"^{key}=([0-9.]+)", text, re.MULTILINE)
            if mm:
                row[key] = float(mm.group(1))
        mm = re.search(r"^pruned_ruu=([0-9.]+)", text, re.MULTILINE)
        if mm:
            row["pruned_utility"] = float(mm.group(1))
        mm = re.search(r"^pruned_oub=([0-9.]+)", text, re.MULTILINE)
        if mm:
            row["pruned_occupancy"] = float(mm.group(1))
    else:
        runtime = re.search(r"Algorithm Core\s+:\s+([0-9.]+)\s+ms", text)
        total = re.search(r"TOTAL WALL TIME\s+:\s+([0-9.]+)\s+ms", text)
        ram = re.search(r"Peak RAM \(VmHWM\)\s+:\s+([0-9.]+)\s+MB", text)
        itemsets = re.search(r"Frequent Itemsets:\s+([0-9]+)", text)
        total_items = re.search(r"Total Items\s+:\s+([0-9]+)", text)
        result_ram = re.search(r"RAM Occupied\s+:\s+[0-9.]+\s+MB\s+\(([0-9]+)\s+Bytes\)", text)
        result_disk = re.search(r"Est\. Disk \(\.txt\)\s+:\s+[0-9.]+\s+MB\s+\(([0-9]+)\s+Bytes\)", text)
        if runtime:
            row["runtime_sec"] = float(runtime.group(1)) / 1000.0
        if total:
            row["total_sec"] = float(total.group(1)) / 1000.0
        if ram:
            row["peak_ram_mb"] = float(ram.group(1))
        if itemsets:
            row["output_count"] = int(itemsets.group(1))
        if total_items:
            row["total_output_items"] = int(total_items.group(1))
        if result_ram:
            row["result_ram_bytes"] = int(result_ram.group(1))
        if result_disk:
            row["result_disk_est_bytes"] = int(result_disk.group(1))
        if row["algorithm"] == "mhoui":
            mm = re.search(r"HOUI=([0-9]+)\s+Weak=([0-9]+)\s+Strong=([0-9]+)", text)
            if mm:
                row["raw_houi_count"] = int(mm.group(1))
                row["weak_count"] = int(mm.group(2))
                row["strong_count"] = int(mm.group(3))
            mm = re.search(r"visited=([0-9]+)", text)
            if mm:
                row["visited_nodes"] = int(mm.group(1))
            mm = re.search(r"pruned_support=([0-9]+)", text)
            if mm:
                row["pruned_support"] = int(mm.group(1))
            mm = re.search(r"pruned_uub=([0-9]+)", text)
            if mm:
                row["pruned_utility"] = int(mm.group(1))
            mm = re.search(r"pruned_oub1=([0-9]+)", text)
            if mm:
                row["pruned_occupancy"] = int(mm.group(1))
    return row


def parse_file(path):
    rows = []
    block = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("===== "):
            if block:
                row = parse_block(block, path)
                if row:
                    rows.append(row)
            block = [line]
        elif block:
            block.append(line)
    if block:
        row = parse_block(block, path)
        if row:
            rows.append(row)
    return rows


def val(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def setting(row):
    return f"u={row['minutil']:.0f}\ns={row['minsup']:.3g}\no={row['minocc']:.2g}"


def write_csv(rows, out):
    keys = ["dataset", "algorithm", "minutil", "minsup", "minocc", "runner_status", "runtime_sec",
            "total_sec", "peak_ram_mb", "output_count", "raw_houi_count", "weak_count", "strong_count",
            "total_output_items", "result_ram_bytes", "result_disk_est_bytes", "visited_nodes",
            "pruned_support", "pruned_utility", "pruned_occupancy", "source_file"]
    with (out / "chuo_q1_grid_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_dataset(rows, out, dataset, metric, title, ylabel, log=False):
    ds_rows = [r for r in rows if r["dataset"] == dataset and r["runner_status"] == "OK"]
    if not ds_rows:
        return
    configs = sorted({(r["minutil"], r["minsup"], r["minocc"]) for r in ds_rows})
    x = range(len(configs))
    plt.figure(figsize=(9.5, 5.0))
    width = 0.16
    for ai, alg in enumerate(ALGORITHMS):
        vals = []
        for cfg in configs:
            hit = next((r for r in ds_rows if r["algorithm"] == alg and (r["minutil"], r["minsup"], r["minocc"]) == cfg), None)
            vals.append(val(hit, metric) if hit else 0.0)
        offset = [(i + (ai - 2) * width) for i in x]
        plt.bar(offset, vals, width=width, label=alg)
    plt.xticks(list(x), [setting({"minutil": c[0], "minsup": c[1], "minocc": c[2]}) for c in configs])
    plt.ylabel(ylabel)
    plt.title(f"{title}: {dataset}")
    if log:
        plt.yscale("log")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / f"{dataset}_{metric}_grid.png", dpi=180)
    plt.close()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/chuo_q1_grid")
    rows = []
    for path in sorted(out.glob("*.txt")):
        rows.extend(parse_file(path))
    rows.sort(key=lambda r: (r["dataset"], ALGORITHMS.index(r["algorithm"]), r["minutil"], r["minsup"], r["minocc"]))
    write_csv(rows, out)
    for ds in sorted({r["dataset"] for r in rows}):
        plot_dataset(rows, out, ds, "runtime_sec", "Runtime over threshold grid", "Seconds", log=True)
        plot_dataset(rows, out, ds, "peak_ram_mb", "Peak RAM over threshold grid", "MB")
        plot_dataset(rows, out, ds, "output_count", "Output size over threshold grid", "Patterns", log=True)
        plot_dataset(rows, out, ds, "result_disk_est_bytes", "Result footprint over threshold grid", "Bytes", log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
