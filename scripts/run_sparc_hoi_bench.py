#!/usr/bin/env python3
import os
import re
import csv
import math
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "dm.exe"
OUT = ROOT / "results" / "sparc_hoi_compare"

DATASETS = [
    {
        "name": "foodmart",
        "path": ROOT / "datasets" / "itemsets" / "foodmartFIM.txt",
        "minsup": 1,
        "thresholds": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    },
    {
        "name": "retail",
        "path": ROOT / "datasets" / "itemsets" / "retail.txt",
        "minsup": 89,
        "thresholds": [0.001, 0.002, 0.003, 0.004, 0.005]
    },
    {
        "name": "mushrooms",
        "path": ROOT / "datasets" / "itemsets" / "mushrooms.txt",
        "minsup": 421,
        "thresholds": [0.05, 0.06, 0.07, 0.08, 0.09]
    },
    {
        "name": "chess",
        "path": ROOT / "datasets" / "itemsets" / "chess.txt",
        "minsup": 3100,
        "thresholds": [0.01, 0.02, 0.03, 0.04, 0.05]
    },
    {
        "name": "connect",
        "path": ROOT / "datasets" / "itemsets" / "connect.txt",
        "minsup": 66000,
        "thresholds": [0.01, 0.02, 0.03, 0.04, 0.05]
    },
    {
        "name": "pumsb",
        "path": ROOT / "datasets" / "itemsets" / "pumsb.txt",
        "minsup": 47000,
        "thresholds": [0.005, 0.010, 0.015, 0.020, 0.025]
    },
    {
        "name": "accidents",
        "path": ROOT / "datasets" / "itemsets" / "accidents.txt",
        "minsup": 320000,
        "thresholds": [0.001, 0.002, 0.003, 0.004, 0.005]
    },
    {
        "name": "T10I4D100K",
        "path": ROOT / "datasets" / "itemsets" / "T10I4D100K.txt",
        "minsup": 1000,
        "thresholds": [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]
    },
    {
        "name": "t20i6d100k",
        "path": ROOT / "datasets" / "itemsets" / "t20i6d100k.txt",
        "minsup": 1000,
        "thresholds": [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]
    }
]

ALGORITHMS = ["sparc_hoi", "aura_hoi", "hep", "dfhoi"]

COUNT_RE = re.compile(
    r"(?:Raw fullset HO itemsets found:|Total high occupancy itemsets found:|Frequent Itemsets:)\s*(\d+)"
)
TIME_RE = re.compile(r"Algorithm Core\s*:\s*([0-9.]+)\s*ms")
RAM_RE = re.compile(r"Peak RAM .*:\s*([0-9.]+)\s*MB")
DISK_RE = re.compile(r"Est. Disk \(.txt\)\s*:\s*([0-9.]+)\s*MB")

def run_cmd(args):
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return proc.stdout, proc.stderr, proc.returncode

def parse_report(stdout):
    itemsets = None
    for line in stdout.splitlines():
        m = COUNT_RE.search(line)
        if m:
            itemsets = int(m.group(1))
            break
    if itemsets is None:
        itemsets = 0

    time_ms = 0.0
    m = TIME_RE.search(stdout)
    if m:
        time_ms = float(m.group(1))

    ram_mb = 0.0
    m = RAM_RE.search(stdout)
    if m:
        ram_mb = float(m.group(1))

    disk_mb = 0.0
    m = DISK_RE.search(stdout)
    if m:
        disk_mb = float(m.group(1))

    return itemsets, time_ms, ram_mb, disk_mb

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_path = ds["path"]
        minsup = ds["minsup"]
        thresholds = ds["thresholds"]

        print(f"\n==========================================")
        print(f"Running benchmarks for dataset: {ds_name}")
        print(f"==========================================")

        for th in thresholds:
            print(f"\nThreshold (min_occupancy) = {th:.6f} (minsup = {minsup})")
            
            algo_metrics = {}

            for algo in ALGORITHMS:
                if algo == "sparc_hoi":
                    args = [str(BIN), "sparc_hoi", str(ds_path), "0", f"{th:.6f}", str(minsup), "0.0", "sum"]
                elif algo == "aura_hoi":
                    args = [str(BIN), "aura_hoi", str(ds_path), "0", f"{th:.6f}", str(minsup), "0.0", "sum", "raw"]
                elif algo == "hep":
                    args = [str(BIN), "hep", str(ds_path), "0", f"{th:.6f}"]
                elif algo == "dfhoi":
                    args = [str(BIN), "dfhoi", str(ds_path), "0", f"{th:.6f}"]
                else:
                    continue

                print(f"  Running {algo:10s}...", end="", flush=True)
                start_t = time.time()
                stdout, stderr, code = run_cmd(args)
                elapsed = time.time() - start_t

                if code != 0:
                    print(f" FAILED (code {code})")
                    print(stderr)
                    continue

                itemsets, time_ms, ram_mb, disk_mb = parse_report(stdout)
                print(f" DONE in {elapsed:.2f}s (found {itemsets} itemsets, {time_ms:.1f}ms core, {ram_mb:.2f}MB RAM)")

                algo_metrics[algo] = {
                    "itemsets": itemsets,
                    "time_ms": time_ms,
                    "ram_mb": ram_mb,
                    "disk_mb": disk_mb
                }

                results.append({
                    "dataset": ds_name,
                    "threshold": th,
                    "minsup": minsup,
                    "algorithm": algo,
                    "itemsets": itemsets,
                    "time_ms": time_ms,
                    "ram_mb": ram_mb,
                    "disk_mb": disk_mb
                })

            counts = [metrics["itemsets"] for metrics in algo_metrics.values()]
            if len(set(counts)) > 1:
                print(f"  [ERROR] Mismatched itemset counts! {dict((k, v['itemsets']) for k, v in algo_metrics.items())}")
            else:
                print(f"  [VERIFIED] All algorithms found exactly {counts[0]} itemsets.")

    # Write summary CSV
    csv_path = OUT / "sparc_hoi_bench_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "threshold", "minsup", "algorithm", "itemsets", "time_ms", "ram_mb", "disk_mb"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary results saved to {csv_path}")

    # Generate Consolidated 3x3 Plots
    print("Generating consolidated line charts (3x3)...")
    metrics_to_plot = [
        ("Runtime", "time_ms", "Runtime (ms)"),
        ("RAM", "ram_mb", "Peak RAM (MB)"),
        ("Disk", "disk_mb", "Estimated Disk (MB)")
    ]

    for metric_name, field, ylabel in metrics_to_plot:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, ds in enumerate(DATASETS):
            ds_name = ds["name"]
            ax = axes[idx]
            
            ds_results = [r for r in results if r["dataset"] == ds_name]
            
            for algo in ALGORITHMS:
                algo_res = [r for r in ds_results if r["algorithm"] == algo]
                algo_res.sort(key=lambda r: r["threshold"])
                
                x = [r["threshold"] for r in algo_res]
                y = [r[field] for r in algo_res]
                
                ax.plot(x, y, marker="o", label=algo)

            ax.set_title(f"{ds_name.capitalize()}")
            ax.set_xlabel("Threshold")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper right")

        plt.suptitle(f"{metric_name} Comparison across All 9 Datasets", fontsize=16, weight="bold")
        plt.tight_layout()
        plot_file = plot_dir / f"all_datasets_{metric_name.lower()}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_file}")

    # Calculate SPARC-HOI Speedup & RAM reduction factors
    print("Calculating overall efficiency factors...")
    speedups = {"aura_hoi": [], "hep": [], "dfhoi": []}
    ram_reductions = {"aura_hoi": [], "hep": [], "dfhoi": []}

    # Group by dataset and threshold
    cases = {}
    for r in results:
        key = (r["dataset"], r["threshold"])
        if key not in cases:
            cases[key] = {}
        cases[key][r["algorithm"]] = r

    for key, algos in cases.items():
        if "sparc_hoi" not in algos:
            continue
        sparc = algos["sparc_hoi"]
        if sparc["time_ms"] <= 0:
            sparc["time_ms"] = 0.1  # avoid division by zero
        if sparc["ram_mb"] <= 0:
            sparc["ram_mb"] = 0.1

        for other in ["aura_hoi", "hep", "dfhoi"]:
            if other in algos:
                o_data = algos[other]
                speedup = o_data["time_ms"] / sparc["time_ms"]
                ram_red = o_data["ram_mb"] / sparc["ram_mb"]
                speedups[other].append(speedup)
                ram_reductions[other].append(ram_red)

    # Plot average factors
    plt.figure(figsize=(8, 5))
    x_labels = ["AURA-HOI", "HEP", "DFHOI"]
    avg_speedups = [np.mean(speedups[algo]) if speedups[algo] else 1.0 for algo in ["aura_hoi", "hep", "dfhoi"]]
    avg_ram_red = [np.mean(ram_reductions[algo]) if ram_reductions[algo] else 1.0 for algo in ["aura_hoi", "hep", "dfhoi"]]

    x = np.arange(len(x_labels))
    width = 0.35

    plt.bar(x - width/2, avg_speedups, width, label="Mean Speedup Factor", color="#1f77b4")
    plt.bar(x + width/2, avg_ram_red, width, label="Mean RAM Reduction Factor", color="#ff7f0e")

    plt.title("SPARC-HOI Overall Efficiency Factors", fontsize=14, weight="bold")
    plt.xticks(x, x_labels)
    plt.ylabel("Factor (Times Better)")
    plt.yscale("log")  # Log scale since memory reduction against HEP is huge
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    
    # Annotate values
    for i, (s, r) in enumerate(zip(avg_speedups, avg_ram_red)):
        plt.text(i - width/2, s * 1.1, f"{s:.1f}x", ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, r * 1.1, f"{r:.1f}x", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    bar_plot_file = plot_dir / "overall_efficiency.png"
    plt.savefig(bar_plot_file, dpi=150)
    plt.close()
    print(f"  Saved plot: {bar_plot_file}")

    print("\nBenchmarking and visualization pipeline completed successfully!")

if __name__ == "__main__":
    main()
