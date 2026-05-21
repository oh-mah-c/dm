#!/usr/bin/env python3
import subprocess
import re
import os
import csv
import time
import random
import math
from pathlib import Path

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_plot = True
except ImportError:
    has_plot = False

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
BIN_HUST = WORKSPACE / "bin" / "hust_tokenize"
BIN_DM = WORKSPACE / "bin" / "dm.exe"
DATA_DIR = WORKSPACE / "datasets" / "synthetic"
OUT_DIR = WORKSPACE / "results" / "hust"
DOC_DIR = WORKSPACE / "docs" / "core"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)

# Signals & Stopwords for Synthetic Generator
STOPWORDS = ["the", "is", "and", "of", "to", "a", "in", "that", "we", "for", "with", "as", "by", "on", "at", "it", "from"]
SIGNALS = ["machine learning", "data mining", "pattern mining", "high utility", "association rules", "frequent patterns"]

def generate_zipf_corpus(filename, num_words, vocab_size, zipf_exponent):
    print(f"Generating Zipfian corpus at {filename} (words={num_words}, V={vocab_size}, s={zipf_exponent})...")
    # Precompute weights
    weights = []
    for r in range(1, vocab_size + 1):
        weights.append(1.0 / (r ** zipf_exponent))
    total = sum(weights)
    cum_weights = []
    curr = 0.0
    for w in weights:
        curr += w
        cum_weights.append(curr / total)
        
    words = [f"w{i}" for i in range(vocab_size)]
    
    chunk_size = 50000
    with open(filename, "w", encoding="utf-8") as f:
        remaining = num_words
        while remaining > 0:
            n = min(remaining, chunk_size)
            sampled = random.choices(words, cum_weights=cum_weights, k=n)
            
            # Inject signals (2-word phrases)
            for i in range(len(sampled) - 2):
                if random.random() < 0.05:
                    sig = random.choice(SIGNALS).split()
                    sampled[i] = sig[0]
                    sampled[i+1] = sig[1]
                    
            f.write(" ".join(sampled) + " ")
            remaining -= n
        f.write("\n")
    print("Corpus generation complete.")

def run_hust(input_file, L, stride, theta_ratio, minsup, max_depth, is_sequence, output_tx_file=None, no_filter=False, max_patterns=25000):
    cmd = [
        str(BIN_HUST),
        "-input", str(input_file),
        "-L", str(L),
        "-d", str(stride),
        "-theta_ratio", f"{theta_ratio:.4f}",
        "-minsup", str(minsup),
        "-max_depth", str(max_depth),
        "-max_patterns", str(max_patterns),
        "--benchmark"
    ]
    if is_sequence:
        cmd += ["--mode", "sequence"]
    if no_filter:
        cmd += ["--no-filter"]
    if output_tx_file:
        cmd += ["-output", str(output_tx_file)]
        
    # Run command
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"HUST execution failed: {res.stderr}")
        return None
        
    # Parse output
    stdout = res.stdout
    metrics = {
        "throughput_mib_s": 0.0,
        "elapsed_time_s": 0.0,
        "peak_rss_kb": 0.0,
        "initial_vocab": 0,
        "active_vocab": 0,
        "discovered_patterns": 0,
        "reduction_ratio": 1.0,
        "visited_nodes": 0,
        "joins": 0
    }
    
    # Extract using regex
    m = re.search(r"Throughput:\s*([0-9.]+)\s*MiB/s", stdout)
    if m: metrics["throughput_mib_s"] = float(m.group(1))
    
    m = re.search(r"Elapsed Time:\s*([0-9.]+)\s*s", stdout)
    if m: metrics["elapsed_time_s"] = float(m.group(1))
    
    m = re.search(r"Peak RSS:\s*([0-9.]+)\s*KB", stdout)
    if m: metrics["peak_rss_kb"] = float(m.group(1))
    
    m = re.search(r"Initial Vocabulary:\s*([0-9]+)", stdout)
    if m: metrics["initial_vocab"] = int(m.group(1))
    
    m = re.search(r"Active Vocabulary .*:\s*([0-9]+)", stdout)
    if m: metrics["active_vocab"] = int(m.group(1))
    
    m = re.search(r"Discovered Patterns:\s*([0-9]+)", stdout)
    if m: metrics["discovered_patterns"] = int(m.group(1))
    
    m = re.search(r"Token Count Reduction Ratio:\s*([0-9.]+)", stdout)
    if m: metrics["reduction_ratio"] = float(m.group(1))
    
    m = re.search(r"Visited Nodes:\s*([0-9]+)", stdout)
    if m: metrics["visited_nodes"] = int(m.group(1))
    
    m = re.search(r"Joins:\s*([0-9]+)", stdout)
    if m: metrics["joins"] = int(m.group(1))
    
    return metrics

def run_dm_fpmax(dataset_file, minsup):
    cmd = [
        str(BIN_DM),
        "fpmax",
        str(dataset_file),
        "0", # Transactional type
        str(minsup)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return None
        
    stdout = res.stdout
    metrics = {
        "algo_core_ms": 0.0,
        "total_wall_time_ms": 0.0,
        "peak_ram_mb": 0.0,
        "output_count": 0
    }
    
    m = re.search(r"Algorithm Core\s*:\s*([0-9.]+)\s*ms", stdout)
    if m: metrics["algo_core_ms"] = float(m.group(1))
    
    m = re.search(r"TOTAL WALL TIME\s*:\s*([0-9.]+)\s*ms", stdout)
    if m: metrics["total_wall_time_ms"] = float(m.group(1))
    
    m = re.search(r"Peak RAM \(VmHWM\)\s*:\s*([0-9.]+)\s*MB", stdout)
    if m: metrics["peak_ram_mb"] = float(m.group(1))
    
    m = re.search(r"Frequent Itemsets:\s*([0-9]+)", stdout)
    if m: metrics["output_count"] = int(m.group(1))
    
    return metrics

def main():
    print("Starting HUST Benchmark Suite...")
    
    # 1. Generate core synthetic dataset for sweeps
    corpus_file = DATA_DIR / "hust_bench_corpus.txt"
    # 300,000 words yields ~2.0 MB corpus, perfect for sweep speed and statistical significance
    generate_zipf_corpus(corpus_file, num_words=300000, vocab_size=5000, zipf_exponent=1.1)
    
    # Sweep configurations
    theta_ratios = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25]
    zipf_skews = [0.8, 1.0, 1.2, 1.4, 1.6]
    minsup_sweep = [2, 5, 10, 15, 20]
    
    # Results containers
    sweep_theta_results = []
    downstream_results = []
    sweep_skew_results = []
    
    # ==========================================
    # Experiment 1: Vary Theta Ratio
    # ==========================================
    print("\n--- Running Experiment 1: Varying Theta Ratio ---")
    for tr in theta_ratios:
        print(f"Running tr={tr}...")
        res = run_hust(corpus_file, L=64, stride=64, theta_ratio=tr, minsup=5, max_depth=3, is_sequence=False)
        if res:
            res["theta_ratio"] = tr
            sweep_theta_results.append(res)
            print(f"  Throughput: {res['throughput_mib_s']} MiB/s, RAM: {res['peak_rss_kb']/1024:.2f} MB, Patterns: {res['discovered_patterns']}, Reduction: {res['reduction_ratio']:.4f}")
            
    # Save sweep theta results to CSV
    csv_theta = OUT_DIR / "sweep_theta.csv"
    with open(csv_theta, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_theta_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_theta_results)
        
    # ==========================================
    # Experiment 2: Downstream Mining Acceleration
    # ==========================================
    print("\n--- Running Experiment 2: Downstream Mining Acceleration ---")
    # We will vary support threshold for both HUST-Tokenize transactions and Baseline transactions
    hust_tx_file = OUT_DIR / "hust_tx.txt"
    baseline_tx_file = OUT_DIR / "baseline_tx.txt"
    
    # Generate HUST transactions (using typical tr=0.15, minsup=5, max_patterns=100)
    print("Generating HUST transactions...")
    hust_res = run_hust(corpus_file, L=64, stride=64, theta_ratio=0.15, minsup=5, max_depth=3, is_sequence=False, output_tx_file=hust_tx_file, max_patterns=100)
    
    # Generate Baseline transactions (no merges, no noise filter)
    print("Generating Baseline transactions...")
    baseline_res = run_hust(corpus_file, L=64, stride=64, theta_ratio=100.0, minsup=5, max_depth=3, is_sequence=False, output_tx_file=baseline_tx_file, no_filter=True)
    
    # Run FPmax on both with varying min_support thresholds (relative ratios)
    minsup_ratios = [0.03, 0.04, 0.05, 0.06, 0.07]
    for ms in minsup_ratios:
        print(f"Running fpmax with minsup={ms}...")
        hust_mine = run_dm_fpmax(hust_tx_file, ms)
        base_mine = run_dm_fpmax(baseline_tx_file, ms)
        
        if hust_mine and base_mine:
            row = {
                "minsup_ratio": ms,
                "hust_core_ms": hust_mine["algo_core_ms"],
                "hust_total_ms": hust_mine["total_wall_time_ms"],
                "hust_patterns": hust_mine["output_count"],
                "base_core_ms": base_mine["algo_core_ms"],
                "base_total_ms": base_mine["total_wall_time_ms"],
                "base_patterns": base_mine["output_count"]
            }
            downstream_results.append(row)
            accel = row["base_core_ms"] / max(row["hust_core_ms"], 0.001)
            print(f"  Minsup {ms:.3f} | Baseline Core: {row['base_core_ms']:.2f} ms | HUST Core: {row['hust_core_ms']:.2f} ms | Acceleration: {accel:.1f}x")
            
    csv_downstream = OUT_DIR / "downstream_acceleration.csv"
    with open(csv_downstream, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=downstream_results[0].keys())
        writer.writeheader()
        writer.writerows(downstream_results)
        
    # ==========================================
    # Experiment 3: Vary Zipf Exponent
    # ==========================================
    print("\n--- Running Experiment 3: Effect of Zipf Skew ---")
    for skew in zipf_skews:
        # Generate temporary corpus with this skew
        skew_corpus = DATA_DIR / f"corpus_skew_{skew:.1f}.txt"
        generate_zipf_corpus(skew_corpus, num_words=150000, vocab_size=3000, zipf_exponent=skew)
        
        print(f"Running with skew={skew}...")
        res = run_hust(skew_corpus, L=64, stride=64, theta_ratio=0.05, minsup=5, max_depth=3, is_sequence=False)
        if res:
            res["skew"] = skew
            sweep_skew_results.append(res)
            print(f"  Throughput: {res['throughput_mib_s']} MiB/s, RAM: {res['peak_rss_kb']/1024:.2f} MB, Patterns: {res['discovered_patterns']}")
            
        # Clean up temp file
        if skew_corpus.exists():
            os.remove(skew_corpus)
            
    csv_skew = OUT_DIR / "sweep_skew.csv"
    with open(csv_skew, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_skew_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_skew_results)
        
    # Plotting
    if has_plot:
        print("\nPlotting results...")
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Color Palette matching premium aesthetics
        c_primary = "#2B5C8F"   # Slate Blue
        c_secondary = "#D95D39" # Terracotta
        c_accent = "#2EC4B6"    # Teal
        c_dark = "#1A1A2E"      # Charcoal
        
        # Plot 1: Throughput and RSS vs Theta Ratio
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()
        
        tr_vals = [r["theta_ratio"] for r in sweep_theta_results]
        tp_vals = [r["throughput_mib_s"] for r in sweep_theta_results]
        ram_vals = [r["peak_rss_kb"] / 1024.0 for r in sweep_theta_results] # Convert to MB
        
        line1 = ax1.plot(tr_vals, tp_vals, color=c_primary, marker='o', linewidth=2.0, label="Throughput (MiB/s)")
        line2 = ax2.plot(tr_vals, ram_vals, color=c_secondary, marker='s', linestyle='--', linewidth=2.0, label="Peak RAM (MB)")
        
        ax1.set_xlabel("Utility Threshold Ratio ($\\theta_{ratio}$)", fontweight='bold')
        ax1.set_ylabel("Throughput (MiB/s)", color=c_primary, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=c_primary)
        
        ax2.set_ylabel("Peak RAM (MB)", color=c_secondary, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=c_secondary)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True)
        
        plt.title("HUST-Tokenize: Throughput & Peak RAM vs. $\\theta_{ratio}$", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(DOC_DIR / "hust_throughput_ram.png", dpi=300)
        plt.close()
        
        # Plot 2: Token Count Reduction Ratio vs Theta Ratio
        plt.figure(figsize=(7, 4.5))
        red_vals = [r["reduction_ratio"] for r in sweep_theta_results]
        plt.plot(tr_vals, red_vals, color=c_accent, marker='^', linewidth=2.5, markersize=8)
        plt.xlabel("Utility Threshold Ratio ($\\theta_{ratio}$)", fontweight='bold')
        plt.ylabel("Token Count Reduction Ratio", fontweight='bold')
        plt.title("HUST-Tokenize: Token Reduction Ratio vs. $\\theta_{ratio}$", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(DOC_DIR / "hust_reduction_ratio.png", dpi=300)
        plt.close()
        
        # Plot 3: Downstream Mining Time Comparison (Baseline vs HUST)
        plt.figure(figsize=(7, 4.5))
        ms_ratios = [r["minsup_ratio"] for r in downstream_results]
        hust_times = [r["hust_core_ms"] for r in downstream_results]
        base_times = [r["base_core_ms"] for r in downstream_results]
        
        plt.plot(ms_ratios, base_times, color=c_secondary, marker='x', linestyle=':', linewidth=2.0, label="Standard Tokenize (Baseline)")
        plt.plot(ms_ratios, hust_times, color=c_primary, marker='o', linewidth=2.5, label="HUST Tokenize")
        plt.xlabel("Downstream FPmax Support Threshold ($minsup$ ratio)", fontweight='bold')
        plt.ylabel("Mining Execution Time (ms)", fontweight='bold')
        plt.yscale('log')
        plt.legend(frameon=True)
        plt.title("Downstream Mining Acceleration: FPmax Performance", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(DOC_DIR / "hust_downstream_acceleration.png", dpi=300)
        plt.close()
        
        # Plot 4: Effect of Zipf Exponent
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()
        
        skew_vals = [r["skew"] for r in sweep_skew_results]
        skew_tp = [r["throughput_mib_s"] for r in sweep_skew_results]
        skew_patt = [r["discovered_patterns"] for r in sweep_skew_results]
        
        line1 = ax1.plot(skew_vals, skew_tp, color=c_primary, marker='o', linewidth=2.0, label="Throughput (MiB/s)")
        line2 = ax2.plot(skew_vals, skew_patt, color=c_accent, marker='^', linestyle='--', linewidth=2.0, label="Discovered Patterns")
        
        ax1.set_xlabel("Zipf Exponent ($s$)", fontweight='bold')
        ax1.set_ylabel("Throughput (MiB/s)", color=c_primary, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=c_primary)
        
        ax2.set_ylabel("Discovered Patterns", color=c_accent, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=c_accent)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True)
        
        plt.title("Effect of Zipf Skew on Tokenization Performance", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(DOC_DIR / "hust_zipf_effect.png", dpi=300)
        plt.close()
        
        print("Plots generated successfully and saved to docs/core/.")
        
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
