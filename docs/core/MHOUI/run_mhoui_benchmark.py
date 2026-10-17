#!/usr/bin/env python3
"""Standardized Multi-Miner Benchmark Runner for MHOUI.

Compares MHOUI-Miner (Direct Single-Phase) against Pluggable Two-Stage Baselines:
  1. MHOUI-Miner (Direct) [Ours]
  2. HOUI + Post-Filter
  3. OCEAN + Post-Filter
  4. HUOPM + Post-Filter
  5. IL-HUOPM + Post-Filter

Across 5 standard real-world benchmark datasets:
  foodmart, liquor, fruithut, chainstore, ecommerce

Saves standardized results to:
  docs/core/MHOUI/mhoui_multiminer_benchmark_results.csv
"""

import csv
import os
import subprocess
import sys
import time

BIN_BENCH = "./bin/mhoui_bench.exe"
OUT_CSV = "docs/core/MHOUI/mhoui_multiminer_benchmark_results.csv"

ALGORITHMS = [
    ("direct_strong_mhoui", "MHOUI-Miner (Direct)"),
    ("houi_filter",         "HOUI + Post-Filter"),
    ("ocean_filter",        "OCEAN + Post-Filter"),
    ("huopm_filter",        "HUOPM + Post-Filter"),
    ("il_huopm_filter",     "IL-HUOPM + Post-Filter")
]

DATASETS = {
    "foodmart": {
        "path": "datasets/utilities/foodmart.txt",
        "minocc": 0.10,
        "minutil_ratio": 0.0001,
        "minsups": [0.002, 0.005, 0.010, 0.015, 0.020]
    },
    "liquor": {
        "path": "datasets/utilities/liquor_11.txt",
        "minocc": 0.20,
        "minutil_ratio": 0.0005,
        "minsups": [0.005, 0.010, 0.015, 0.020, 0.030]
    },
    "fruithut": {
        "path": "datasets/utilities/fruithut_utility.txt",
        "minocc": 0.20,
        "minutil_ratio": 0.0010,
        "minsups": [0.005, 0.010, 0.015, 0.020, 0.030]
    },
    "chainstore": {
        "path": "datasets/utilities/chainstore.txt",
        "minocc": 0.20,
        "minutil_ratio": 0.0020,
        "minsups": [0.005, 0.010, 0.015, 0.020, 0.030]
    },
    "ecommerce": {
        "path": "datasets/utilities/ecommerce_utility_no_timestamps.txt",
        "minocc": 0.10,
        "minutil_ratio": 0.0005,
        "minsups": [0.002, 0.005, 0.008, 0.010, 0.020]
    }
}


def run_single(ds_path, algo_id, minsup, minocc, u_ratio):
    cmd = [
        BIN_BENCH,
        "--input", ds_path,
        "--algorithm", algo_id,
        "--minsup", str(minsup),
        "--minocc", str(minocc),
        "--minutil-ratio", str(u_ratio),
        "--strong", "1"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            return None

        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 8 and parts[0] == algo_id:
                return {
                    "time_mine_s": float(parts[1]),
                    "time_filter_s": float(parts[2]),
                    "time_total_s": float(parts[3]),
                    "visited_nodes": int(parts[4]),
                    "candidates": int(parts[5]),
                    "patterns": int(parts[6]),
                    "peak_ram_mb": float(parts[7])
                }
    except Exception as e:
        print(f"    Exception: {e}")
        return None
    return None


def main():
    os.makedirs("docs/core/MHOUI", exist_ok=True)
    print("=" * 90)
    print("MHOUI MULTI-MINER COMPARATIVE SCALABILITY BENCHMARK")
    print("=" * 90)

    rows = []

    for ds_name, ds_info in DATASETS.items():
        ds_path = ds_info["path"]
        minocc = ds_info["minocc"]
        u_ratio = ds_info["minutil_ratio"]
        print(f"\n--- Dataset: {ds_name} ({ds_path}) | minocc={minocc}, minutil_ratio={u_ratio} ---")

        for minsup in ds_info["minsups"]:
            print(f"  Threshold minsup = {minsup}:")
            for algo_id, algo_desc in ALGORITHMS:
                res = run_single(ds_path, algo_id, minsup, minocc, u_ratio)
                if not res:
                    print(f"    {algo_desc:32s} -> ERROR / TIMEOUT")
                    continue

                print(f"    {algo_desc:32s} -> Time: {res['time_total_s']:7.4f}s (mine={res['time_mine_s']:.4f}s, filter={res['time_filter_s']:.4f}s) | "
                      f"Nodes: {res['visited_nodes']:8d} | Pats: {res['patterns']:6d} | RAM: {res['peak_ram_mb']:6.1f}MB")

                rows.append({
                    "dataset": ds_name,
                    "algorithm": algo_id,
                    "description": algo_desc,
                    "minsup": minsup,
                    "minocc": minocc,
                    "minutil_ratio": u_ratio,
                    "time_mine_s": res["time_mine_s"],
                    "time_filter_s": res["time_filter_s"],
                    "time_total_s": res["time_total_s"],
                    "visited_nodes": res["visited_nodes"],
                    "candidates": res["candidates"],
                    "patterns": res["patterns"],
                    "peak_ram_mb": res["peak_ram_mb"]
                })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 90)
    print(f"Benchmark completed successfully. Results saved to: {OUT_CSV}")
    print("=" * 90)


if __name__ == "__main__":
    main()
