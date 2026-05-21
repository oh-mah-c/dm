#!/usr/bin/env python3
import subprocess
import re
import os
import time
import shutil
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
BIN_DM = WORKSPACE / "bin" / "dm.exe"
OUT_DIR = WORKSPACE / "results" / "medm_gen"
DOC_DIR = WORKSPACE / "docs" / "core"
ARTIFACT_DIR = Path("/home/autocookie/.gemini/antigravity/brain/c269e029-388f-4471-8c1a-d1a4e734c91e")

OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)

def write_spec(modality, size, threshold, epsilon=5.0, max_iter=20):
    spec_path = OUT_DIR / f"temp_{modality}_spec.ini"
    content = f"""# MeDM-Gen experimental spec
modality = {modality}
size = {size}
item_count = 1000
avg_len = 10
length_dist = poisson
noise_rate = 0.05
epsilon = {epsilon}
max_len = 25
allow_duplicates = false
support_error_threshold = {threshold}
max_iterations = {max_iter}

# Planted patterns
pattern = 10,20:0.25
pattern = 30,40,50:0.15
pattern = 60,70,80,90:0.10
"""
    with open(spec_path, "w") as f:
        f.write(content)
    return spec_path

def run_generator(spec_path, output_base):
    cmd = [
        str(BIN_DM),
        "medm_gen",
        str(spec_path),
        str(output_base),
        "",
        "42"
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    
    if res.returncode != 0:
        print(f"Generator failed: {res.stderr}")
        return None, 0.0
        
    # Parse output to extract iterations and deviations
    deviations = []
    lines = res.stdout.splitlines()
    for line in lines:
        m = re.search(r"Max Support Deviation = ([0-9.]+)", line)
        if m:
            deviations.append(float(m.group(1)))
            
    return deviations, (t1 - t0) * 1000.0 # runtime in ms

def run_experiments():
    print("Starting MeDM-Gen Empirical Evaluation...")
    
    # 1. Convergence Experiment (Deviation vs Iteration for varying size N)
    print("Running Convergence Experiment...")
    convergence_data = {}
    sizes = [1000, 10000, 100000]
    for size in sizes:
        spec_path = write_spec("transactional", size, threshold=0.001, max_iter=15)
        deviations, _ = run_generator(spec_path, OUT_DIR / f"out_conv_{size}")
        if deviations:
            convergence_data[size] = deviations
            print(f"  N={size}: Mapped deviations over {len(deviations)} iterations: {deviations}")
            
    # 2. Runtime Experiment (Runtime vs size N for Transactional, Utility, Sequence)
    print("Running Runtime Experiment...")
    runtime_data = {"transactional": [], "utility": [], "sequence": []}
    run_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
    for modality in ["transactional", "utility", "sequence"]:
        for size in run_sizes:
            # use loose threshold to avoid iteration overhead during runtime benchmarking
            spec_path = write_spec(modality, size, threshold=0.05, max_iter=1)
            _, runtime_ms = run_generator(spec_path, OUT_DIR / f"out_run_{modality}_{size}")
            runtime_data[modality].append(runtime_ms)
            print(f"  Modality={modality}, N={size}: Runtime = {runtime_ms:.2f} ms")

    # 3. Privacy vs. Utility Experiment (Deviation vs. Epsilon DP)
    print("Running Privacy vs. Utility Experiment...")
    privacy_data = []
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for eps in epsilons:
        spec_path = write_spec("transactional", 50000, threshold=0.005, epsilon=eps, max_iter=5)
        deviations, _ = run_generator(spec_path, OUT_DIR / f"out_priv_{eps}")
        if deviations:
            final_dev = deviations[-1]
            privacy_data.append(final_dev)
            print(f"  Epsilon={eps}: Final Support Deviation = {final_dev:.4f}")
            
    # Plotting
    if has_plot:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Plot 1: Convergence
        plt.figure(figsize=(6, 4))
        for size, devs in convergence_data.items():
            plt.plot(range(1, len(devs) + 1), devs, marker='o', label=f'N = {size:,}')
        plt.xlabel("Feedback Loop Iterations")
        plt.ylabel("Max Support Deviation")
        plt.title("Constraint Repair Convergence vs. Dataset Size")
        plt.legend()
        plt.tight_layout()
        conv_png = DOC_DIR / "medm_gen_convergence.png"
        plt.savefig(conv_png, dpi=300)
        plt.close()
        
        # Plot 2: Runtime
        plt.figure(figsize=(6, 4))
        for modality, runtimes in runtime_data.items():
            plt.plot(run_sizes, runtimes, marker='s', label=modality.capitalize())
        plt.xlabel("Dataset Size (Number of Records)")
        plt.ylabel("Generation Time (ms)")
        plt.title("Generator Execution Time vs. Dataset Size")
        plt.legend()
        plt.tight_layout()
        run_png = DOC_DIR / "medm_gen_runtime.png"
        plt.savefig(run_png, dpi=300)
        plt.close()
        
        # Plot 3: Privacy vs Utility
        plt.figure(figsize=(6, 4))
        plt.plot(epsilons, privacy_data, marker='^', color='crimson', linestyle='-', linewidth=2)
        plt.xlabel("Privacy Budget Epsilon (ε)")
        plt.ylabel("Final Max Support Deviation")
        plt.title("Support Deviation vs. Differential Privacy Epsilon")
        plt.xscale('log')
        plt.tight_layout()
        priv_png = DOC_DIR / "medm_gen_privacy.png"
        plt.savefig(priv_png, dpi=300)
        plt.close()
        
        print("Generated all experiment charts.")
        
        # Copy to artifact directory for model walkthrough rendering
        if ARTIFACT_DIR.exists():
            shutil.copy(conv_png, ARTIFACT_DIR / "medm_gen_convergence.png")
            shutil.copy(run_png, ARTIFACT_DIR / "medm_gen_runtime.png")
            shutil.copy(priv_png, ARTIFACT_DIR / "medm_gen_privacy.png")
            print("Copied charts to artifact directory.")
            
    else:
        print("Matplotlib not available. No charts plotted.")

if __name__ == "__main__":
    run_experiments()
