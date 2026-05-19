import os
import time
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

# Styles for professional look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
})

# Path configurations
BASE_DIR = "/home/autocookie/pomaieco/dm"
BIN_PATH = os.path.join(BASE_DIR, "src/tokenizer/bin/tokenizer_miner")
DATASET_PATH = os.path.join(BASE_DIR, "datasets/Malware_types_in_SPMF_Format/Adwaretranslated.txt")
OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load the dataset into memory for Python tokenizers
print("Reading dataset into memory...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()
file_size_mib = os.path.getsize(DATASET_PATH) / (1024 * 1024)
print(f"Dataset Loaded: {file_size_mib:.2f} MiB")

# 2. Setup HuggingFace Tokenizer
print("Loading HuggingFace GPT-2 BPE Tokenizer...")
from tokenizers import Tokenizer
hf_tokenizer = Tokenizer.from_pretrained("gpt2")

# 3. Setup OpenAI Tiktoken
print("Loading OpenAI Tiktoken (cl100k_base)...")
import tiktoken
tiktoken_enc = tiktoken.get_encoding("cl100k_base")

# 4. Benchmark HuggingFace GPT-2 BPE
print("Benchmarking HuggingFace GPT-2 BPE (5 runs)...")
hf_throughputs = []
hf_token_count = 0
for i in range(5):
    t0 = time.perf_counter()
    tokens = hf_tokenizer.encode(text)
    t1 = time.perf_counter()
    hf_token_count = len(tokens.ids)
    hf_throughputs.append(file_size_mib / (t1 - t0))
hf_median = np.median(hf_throughputs)
print(f"  HuggingFace Throughput: {hf_median:.2f} MiB/s | Generated {hf_token_count} tokens")

# 5. Benchmark OpenAI Tiktoken
print("Benchmarking OpenAI Tiktoken (5 runs)...")
tik_throughputs = []
tik_token_count = 0
for i in range(5):
    t0 = time.perf_counter()
    tokens = tiktoken_enc.encode(text)
    t1 = time.perf_counter()
    tik_token_count = len(tokens)
    tik_throughputs.append(file_size_mib / (t1 - t0))
tik_median = np.median(tik_throughputs)
print(f"  OpenAI Tiktoken Throughput: {tik_median:.2f} MiB/s | Generated {tik_token_count} tokens")

# 6. Benchmark FARO-Tokenizer (RH-Arena)
print("Benchmarking FARO-Tokenizer (RH-Arena) (5 runs)...")
faro_arena_throughputs = []
faro_unique = 0
for i in range(5):
    cmd = [BIN_PATH, "-i", DATASET_PATH, "-m", "doc", "-o", "/dev/null", "--benchmark", "--algo", "rh-arena", "--json"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    metrics = json.loads(res.stdout.strip())
    faro_arena_throughputs.append(metrics["throughput_mib"])
    faro_unique = metrics["unique_tokens"]
faro_arena_median = np.median(faro_arena_throughputs)
print(f"  FARO RH-Arena Throughput: {faro_arena_median:.2f} MiB/s | Generated {faro_unique} unique lexemes")

# 7. Benchmark FARO-Tokenizer (RH-Borrow)
print("Benchmarking FARO-Tokenizer (RH-Borrow) (5 runs)...")
faro_borrow_throughputs = []
for i in range(5):
    cmd = [BIN_PATH, "-i", DATASET_PATH, "-m", "doc", "-o", "/dev/null", "--benchmark", "--algo", "rh-borrow", "--json"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    metrics = json.loads(res.stdout.strip())
    faro_borrow_throughputs.append(metrics["throughput_mib"])
faro_borrow_median = np.median(faro_borrow_throughputs)
print(f"  FARO RH-Borrow Throughput: {faro_borrow_median:.2f} MiB/s")

# Plot Results
print("Generating figure4_industry_compare.png...")
fig, ax = plt.subplots(figsize=(8, 5))

labels = ["HF GPT-2 (Rust/BPE)", "Tiktoken (Rust/BPE)", "FARO RH-Arena (C)", "FARO RH-Borrow (Zero-Copy C)"]
throughputs = [hf_median, tik_median, faro_arena_median, faro_borrow_median]
colors = ["#90a4ae", "#78909c", "#ff9800", "#e65100"]

rects = ax.bar(labels, throughputs, color=colors, edgecolor='none', width=0.5)

# Add values on top of bars
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', color='#37474f')

ax.set_ylabel('Throughput (MiB/s)', fontweight='bold')
ax.set_title('Tokenization Throughput: FARO vs. Industry Standard Tokenizers', fontweight='bold', pad=15)
ax.set_ylim(0, max(throughputs) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure4_industry_compare.png"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "figure4_industry_compare.svg"), format='svg')
plt.close()

print("==========================================================")
print(f"Successfully generated Figure 4 in '{OUT_DIR}':")
print("  - figure4_industry_compare.png / .svg")
print("==========================================================")
