import subprocess
import time
import re
import csv
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path("docs/core/MFHOI")
CSV_PATH = OUTDIR / "mfhoi_benchmark_results.csv"

# Datasets and test grid across 5 diverse benchmarks:
# (dataset_name, file_path, total_trans, beta, [(alpha, minsup), ...])
BENCH_PLAN = [
    ("chess", "datasets/itemsets/chess.txt", 3196, 0.40, [
        (0.50, 1598),
        (0.55, 1758),
        (0.60, 1918),
        (0.65, 2078),
        (0.70, 2238),
    ]),
    ("mushrooms", "datasets/itemsets/mushrooms.txt", 8416, 0.35, [
        (0.15, 1263),
        (0.20, 1684),
        (0.25, 2104),
        (0.30, 2525),
        (0.35, 2946),
    ]),
    ("pumsb", "datasets/itemsets/pumsb.txt", 49046, 0.50, [
        (0.80, 39237),
        (0.825, 40463),
        (0.85, 41690),
        (0.875, 42916),
        (0.90, 44142),
    ]),
    ("retail", "datasets/itemsets/retail.txt", 88162, 0.20, [
        (0.005, 441),
        (0.010, 882),
        (0.015, 1323),
        (0.020, 1764),
        (0.030, 2645),
    ]),
    ("T10I4D100K", "datasets/itemsets/T10I4D100K.txt", 100000, 0.20, [
        (0.005, 500),
        (0.010, 1000),
        (0.015, 1500),
        (0.020, 2000),
        (0.030, 3000),
    ]),
]

# Baseline algorithms without AURA-HOI:
# MFHOI-Miner (Ours), GenMax (MFI), MAFIA (MFI), LCM (Closed FCI), HEP (Raw HOI), DFHOI (Raw HOI)
ALGORITHMS = [
    "mfhoi",
    "genmax",
    "mafia",
    "lcm",
    "hep",
    "dfhoi",
]

def run_command(cmd, timeout_s=45):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
        wall_t = time.time() - t0
        core_m = re.search(r"Algorithm Core\s*:\s*([0-9.]+)\s*ms", p.stdout)
        core_t = float(core_m.group(1)) / 1000.0 if core_m else wall_t
        ram_m = re.search(r"Peak RAM \(VmHWM\)\s*:\s*([0-9.]+)\s*MB", p.stdout)
        ram = float(ram_m.group(1)) if ram_m else 0.0
        
        pat = 0
        if "Output=" in p.stdout:
            m = re.search(r"Output=(\d+)", p.stdout)
            if m: pat = int(m.group(1))
        elif "Frequent Itemsets:" in p.stdout:
            m = re.search(r"Frequent Itemsets:\s*(\d+)", p.stdout)
            if m: pat = int(m.group(1))
        elif "Total frequent closed itemsets found:" in p.stdout:
            m = re.search(r"Total frequent closed itemsets found:\s*(\d+)", p.stdout)
            if m: pat = int(m.group(1))
        elif "Total maximal frequent itemsets found:" in p.stdout:
            m = re.search(r"Total maximal frequent itemsets found:\s*(\d+)", p.stdout)
            if m: pat = int(m.group(1))
        elif "itemsets found:" in p.stdout:
            m = re.search(r"itemsets found:\s*(\d+)", p.stdout)
            if m: pat = int(m.group(1))
            
        return core_t, ram, pat, False
    except subprocess.TimeoutExpired:
        return timeout_s, 0.0, 0, True
    except Exception as e:
        return 0.0, 0.0, 0, True

def main():
    rows = []
    print(f"Starting benchmark across {len(BENCH_PLAN)} datasets and {len(ALGORITHMS)} algorithms (NO AURA-HOI)...")

    for ds_name, path, n_trans, beta, grid in BENCH_PLAN:
        print(f"\n==========================================")
        print(f"Dataset: {ds_name} (n={n_trans}, beta={beta})")
        print(f"==========================================")
        for alpha, minsup in grid:
            for algo in ALGORITHMS:
                if algo == "mfhoi":
                    cmd = f"./build/bin/dm.exe mfhoi {path} 0 {alpha} {beta} 1"
                elif algo == "genmax":
                    cmd = f"./build/bin/dm.exe genmax {path} 0 {minsup}"
                elif algo == "mafia":
                    cmd = f"./build/bin/dm.exe mafia {path} 0 {minsup}"
                elif algo == "lcm":
                    cmd = f"./build/bin/dm.exe lcm {path} 0 {minsup}"
                elif algo == "hep":
                    cmd = f"./build/bin/dm.exe hep {path} 0 {alpha} {minsup}"
                elif algo == "dfhoi":
                    cmd = f"./build/bin/dm.exe dfhoi {path} 0 {alpha} {minsup}"
                else:
                    continue

                core_t, ram, pat, timed_out = run_command(cmd)
                status = "TIMEOUT" if timed_out else "OK"
                print(f"  [{ds_name}] α={alpha:<5} minsup={minsup:<5} | {algo:<10} | time: {core_t:8.4f}s | ram: {ram:6.1f}MB | pat: {pat:<8} | {status}")
                rows.append({
                    "dataset": ds_name,
                    "algorithm": algo,
                    "alpha": alpha,
                    "beta": beta,
                    "minsup": minsup,
                    "time_s": core_t,
                    "peak_ram_mb": ram,
                    "patterns": pat,
                    "timed_out": timed_out
                })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nBenchmark completed! Raw results saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
