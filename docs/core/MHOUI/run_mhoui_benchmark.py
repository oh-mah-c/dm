#!/usr/bin/env python3
import subprocess
import time
import os
import csv
import sys

DATASETS = {
    "foodmart": {
        "path": "datasets/utilities/foodmart.txt",
        "grid": [
            {"minsup": 0.005, "minocc": 0.10, "minutil_ratio": 0.0001},
            {"minsup": 0.005, "minocc": 0.20, "minutil_ratio": 0.0005},
            {"minsup": 0.010, "minocc": 0.10, "minutil_ratio": 0.0001},
            {"minsup": 0.010, "minocc": 0.20, "minutil_ratio": 0.0005},
            {"minsup": 0.020, "minocc": 0.10, "minutil_ratio": 0.0005},
            {"minsup": 0.020, "minocc": 0.20, "minutil_ratio": 0.0010},
        ]
    },
    "liquor": {
        "path": "datasets/utilities/liquor_11.txt",
        "grid": [
            {"minsup": 0.010, "minocc": 0.20, "minutil_ratio": 0.0005},
            {"minsup": 0.010, "minocc": 0.30, "minutil_ratio": 0.0010},
            {"minsup": 0.020, "minocc": 0.20, "minutil_ratio": 0.0005},
            {"minsup": 0.020, "minocc": 0.30, "minutil_ratio": 0.0010},
            {"minsup": 0.030, "minocc": 0.20, "minutil_ratio": 0.0010},
            {"minsup": 0.050, "minocc": 0.20, "minutil_ratio": 0.0010},
        ]
    },
    "fruithut": {
        "path": "datasets/utilities/fruithut_utility.txt",
        "grid": [
            {"minsup": 0.005, "minocc": 0.20, "minutil_ratio": 0.0005},
            {"minsup": 0.010, "minocc": 0.20, "minutil_ratio": 0.0010},
            {"minsup": 0.010, "minocc": 0.30, "minutil_ratio": 0.0010},
            {"minsup": 0.020, "minocc": 0.20, "minutil_ratio": 0.0010},
            {"minsup": 0.030, "minocc": 0.20, "minutil_ratio": 0.0010},
            {"minsup": 0.050, "minocc": 0.20, "minutil_ratio": 0.0010},
        ]
    },
    "chainstore": {
        "path": "datasets/utilities/chainstore.txt",
        "grid": [
            {"minsup": 0.010, "minocc": 0.20, "minutil_ratio": 0.0020},
            {"minsup": 0.010, "minocc": 0.30, "minutil_ratio": 0.0050},
            {"minsup": 0.020, "minocc": 0.20, "minutil_ratio": 0.0020},
            {"minsup": 0.020, "minocc": 0.30, "minutil_ratio": 0.0050},
            {"minsup": 0.030, "minocc": 0.20, "minutil_ratio": 0.0050},
            {"minsup": 0.050, "minocc": 0.20, "minutil_ratio": 0.0050},
        ]
    },
    "ecommerce": {
        "path": "datasets/utilities/ecommerce_utility_no_timestamps.txt",
        "grid": [
            {"minsup": 0.005, "minocc": 0.10, "minutil_ratio": 0.0005},
            {"minsup": 0.005, "minocc": 0.20, "minutil_ratio": 0.0010},
            {"minsup": 0.010, "minocc": 0.10, "minutil_ratio": 0.0005},
            {"minsup": 0.010, "minocc": 0.20, "minutil_ratio": 0.0010},
            {"minsup": 0.020, "minocc": 0.10, "minutil_ratio": 0.0010},
            {"minsup": 0.020, "minocc": 0.20, "minutil_ratio": 0.0020},
        ]
    }
}

ALGORITHMS = [
    "houi",
    "weak_mhoui",
    "strong_mhoui",
    "direct_weak_mhoui",
    "direct_strong_mhoui"
]

OUT_CSV = "docs/core/MHOUI/mhoui_benchmark_results.csv"

def run_bench():
    os.makedirs("docs/core/MHOUI", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    fields = [
        "dataset", "algorithm", "minsup", "minocc", "minutil_ratio",
        "runtime_sec", "peak_ram_mb", "visited_nodes", "candidates",
        "pruned_support", "pruned_twu", "pruned_uub", "pruned_oub1", "pruned_oub2", "pruned_dom",
        "num_houi", "num_weak_mhoui", "num_strong_mhoui", "output_count",
        "avg_len", "max_len", "avg_support", "avg_utility", "avg_occupancy"
    ]
    
    rows = []
    
    for ds_name, ds_info in DATASETS.items():
        path = ds_info["path"]
        print(f"\n=================== Dataset: {ds_name} ({path}) ===================")
        for cfg in ds_info["grid"]:
            minsup = cfg["minsup"]
            minocc = cfg["minocc"]
            u_ratio = cfg["minutil_ratio"]
            
            for algo in ALGORITHMS:
                cmd = [
                    "./build/bin/mhoui_miner",
                    "--input", path,
                    "--algorithm", algo,
                    "--minsup", str(minsup),
                    "--minocc", str(minocc),
                    "--minutil-ratio", str(u_ratio),
                    "--output", f"results/bench_{ds_name}_{algo}.csv"
                ]
                
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    out_text = res.stdout
                    
                    data = {
                        "dataset": ds_name,
                        "algorithm": algo,
                        "minsup": minsup,
                        "minocc": minocc,
                        "minutil_ratio": u_ratio,
                    }
                    
                    for line in out_text.strip().split("\n"):
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            if k in fields:
                                try:
                                    if "." in v or "e" in v.lower():
                                        data[k] = float(v)
                                    else:
                                        data[k] = int(v)
                                except ValueError:
                                    data[k] = v
                    
                    rows.append(data)
                    print(f"[{ds_name}] {algo:20s} sup={minsup:<5} occ={minocc:<4} util={u_ratio:<6} -> "
                          f"time={data.get('runtime_sec', 0):.4f}s RAM={data.get('peak_ram_mb', 0):.1f}MB "
                          f"out={data.get('output_count', 0)}")
                except subprocess.TimeoutExpired:
                    print(f"[{ds_name}] {algo:20s} TIMEOUT (>120s)")
                except Exception as e:
                    print(f"[{ds_name}] {algo:20s} ERROR: {e}")
                    
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
            
    print(f"\nSaved {len(rows)} benchmark records to {OUT_CSV}")

if __name__ == "__main__":
    run_bench()
