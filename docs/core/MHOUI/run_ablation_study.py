#!/usr/bin/env python3
"""Ablation Study Runner for MHOUI-Miner.

Quantitatively evaluates the isolated contribution of each algorithmic component:
  1. Full MHOUI-Miner:
       Dual bounds (UBjoint, OUB) + Pareto-Trie Branch Dominance Pruning (Direct Mining)
  2. Ablation (No Branch Dominance):
       disable_dominance = 1 (removes Pareto-Trie superset dominance pruning)
  3. Ablation (No Occupancy Envelope Bound):
       disable_oub = 1 (removes OUB pruning)
  4. Ablation (No Utility Upper Bound):
       disable_uub = 1 (removes UBjoint pruning)
  5. Ablation (No Joint Bounds):
       disable_oub = 1, disable_uub = 1
  6. Two-Stage Baseline (HOUI-Miner + Pluggable Post-Filter):
       Full HOUI enumeration without branch dominance, followed by candidate post-filtering.

Saves benchmark metrics to:
  docs/core/MHOUI/ablation_benchmark_results.csv
"""

import csv
import math
import os
import subprocess
import sys
import time

BIN_BENCH = "./bin/mhoui_bench.exe"
OUT_CSV = "docs/core/MHOUI/ablation_benchmark_results.csv"

ABLATION_MODES = [
    ("direct_strong_mhoui", "Full MHOUI-Miner (All Pruning)"),
    ("ablation_no_dom",     "w/o Branch Pareto Dominance"),
    ("ablation_no_oub",     "w/o Occupancy Envelope (OUB)"),
    ("ablation_no_uub",     "w/o Utility Bound (UBjoint)"),
    ("ablation_no_bounds",  "w/o Joint Bounds (Neither)"),
    ("houi_filter",         "Two-Stage Baseline (HOUI + Post-Filter)")
]

DATASETS = {
    "foodmart": {
        "path": "datasets/utilities/foodmart.txt",
        "configs": [
            {"minsup": 0.001, "minocc": 0.10, "minutil_ratio": 0.001},
            {"minsup": 0.002, "minocc": 0.20, "minutil_ratio": 0.002},
        ]
    },
    "liquor": {
        "path": "datasets/utilities/liquor_11.txt",
        "configs": [
            {"minsup": 0.003, "minocc": 0.20, "minutil_ratio": 0.003},
            {"minsup": 0.005, "minocc": 0.20, "minutil_ratio": 0.005},
        ]
    },
    "fruithut": {
        "path": "datasets/utilities/fruithut_utility.txt",
        "configs": [
            {"minsup": 0.002, "minocc": 0.20, "minutil_ratio": 0.005},
            {"minsup": 0.005, "minocc": 0.20, "minutil_ratio": 0.010},
        ]
    },
    "chainstore": {
        "path": "datasets/utilities/chainstore.txt",
        "configs": [
            {"minsup": 0.001, "minocc": 0.20, "minutil_ratio": 0.005},
            {"minsup": 0.002, "minocc": 0.30, "minutil_ratio": 0.010},
        ]
    },
    "ecommerce": {
        "path": "datasets/utilities/ecommerce_utility_no_timestamps.txt",
        "configs": [
            {"minsup": 0.002, "minocc": 0.20, "minutil_ratio": 0.003},
            {"minsup": 0.005, "minocc": 0.20, "minutil_ratio": 0.005},
        ]
    }
}


def run_single(ds_path, algo, minsup, minocc, u_ratio):
    cmd = [
        BIN_BENCH,
        "--input", ds_path,
        "--algorithm", algo,
        "--minsup", str(minsup),
        "--minocc", str(minocc),
        "--minutil-ratio", str(u_ratio),
        "--strong", "1"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None

    # Parse table row
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 8 and parts[0] == algo:
            return {
                "time_mine_s": float(parts[1]),
                "time_filter_s": float(parts[2]),
                "time_total_s": float(parts[3]),
                "visited_nodes": int(parts[4]),
                "candidates": int(parts[5]),
                "maximal": int(parts[6]),
                "peak_ram_mb": float(parts[7])
            }
    return None


def main():
    os.makedirs("docs/core/MHOUI", exist_ok=True)
    print("=" * 90)
    print("MHOUI-MINER ABLATION STUDY: COMPONENT-WISE PRUNING & EFFICIENCY EVALUATION")
    print("=" * 90)

    rows = []

    for ds_name, ds_info in DATASETS.items():
        ds_path = ds_info["path"]
        print(f"\n--- Dataset: {ds_name} ({ds_path}) ---")

        for cfg_idx, cfg in enumerate(ds_info["configs"]):
            minsup = cfg["minsup"]
            minocc = cfg["minocc"]
            u_ratio = cfg["minutil_ratio"]
            print(f"  Config {cfg_idx + 1}: minsup={minsup}, minocc={minocc}, minutil_ratio={u_ratio}")

            baseline_candidates = None
            for algo_id, algo_desc in ABLATION_MODES:
                res = run_single(ds_path, algo_id, minsup, minocc, u_ratio)
                if not res:
                    print(f"    {algo_desc:45s} -> ERROR")
                    continue

                if algo_id == "houi_filter":
                    baseline_candidates = res["candidates"]

                cand_pruned_pct = 0.0
                if baseline_candidates and baseline_candidates > 0 and res["candidates"] <= baseline_candidates:
                    cand_pruned_pct = 100.0 * (1.0 - res["candidates"] / baseline_candidates)

                print(f"    {algo_desc:45s} -> Time: {res['time_total_s']:7.4f}s | Nodes: {res['visited_nodes']:7d} | Cands: {res['candidates']:8d} | Max: {res['maximal']:6d} | RAM: {res['peak_ram_mb']:6.1f}MB")

                rows.append({
                    "dataset": ds_name,
                    "config_id": cfg_idx + 1,
                    "minsup": minsup,
                    "minocc": minocc,
                    "minutil_ratio": u_ratio,
                    "algorithm": algo_id,
                    "description": algo_desc,
                    "time_mine_s": res["time_mine_s"],
                    "time_filter_s": res["time_filter_s"],
                    "time_total_s": res["time_total_s"],
                    "visited_nodes": res["visited_nodes"],
                    "candidates": res["candidates"],
                    "maximal": res["maximal"],
                    "peak_ram_mb": res["peak_ram_mb"]
                })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 90)
    print(f"Ablation study completed. Results saved to: {OUT_CSV}")
    print("=" * 90)


if __name__ == "__main__":
    main()
