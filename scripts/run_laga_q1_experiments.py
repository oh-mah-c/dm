#!/usr/bin/env python3
import csv
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "dm.exe"
OUT = ROOT / "results" / "laga_q1"
CHARTS = OUT / "charts"
TMP = OUT / "tmp"


def read_records(path, limit=0):
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                line = line.split(":", 1)[0]
            toks = tuple(tok for tok in re.split(r"[\s,]+", line) if tok)
            if toks:
                records.append(toks)
            if limit and len(records) >= limit:
                break
    return records


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(" ".join(map(str, r)) + "\n")


def mine_patterns(records, minsup):
    n = len(records)
    if n == 0:
        return {}
    threshold = math.ceil(minsup * n) if minsup < 1 else int(minsup)
    cnt = Counter()
    for rec in records:
        uniq = sorted(set(rec))
        for a in uniq:
            cnt[(a,)] += 1
        for a, b in combinations(uniq, 2):
            cnt[(a, b)] += 1
    return {p: c / n for p, c in cnt.items() if c >= threshold}


def pattern_metrics(real_records, syn_records, minsup):
    real = mine_patterns(real_records, minsup)
    syn = mine_patterns(syn_records, minsup)
    real_set, syn_set = set(real), set(syn)
    inter = real_set & syn_set
    recall = len(inter) / len(real_set) if real_set else 1.0
    precision = len(inter) / len(syn_set) if syn_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support_dev = sum(abs(real[p] - syn.get(p, 0.0)) for p in real) / len(real) if real else 0.0
    return {
        "real_count": len(real),
        "syn_count": len(syn),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "support_dev": support_dev,
        "real_patterns": real,
        "syn_patterns": syn,
    }


def nearest_similarity(records_a, records_b):
    vals = []
    b_sets = [set(r) for r in records_b]
    for r in records_a:
        s = set(r)
        best = 0.0
        for t in b_sets:
            u = len(s | t)
            if u:
                best = max(best, len(s & t) / u)
        vals.append(best)
    return vals


def parse_stats(stdout, stats_path=None):
    data = {}
    for text in [stdout, Path(stats_path).read_text(errors="ignore") if stats_path and Path(stats_path).exists() else ""]:
        for line in text.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                data[k.strip()] = float(v.strip())
            except ValueError:
                data[k.strip()] = v.strip()
    return data


def run_cmd(cmd, measure_memory=False):
    t0 = time.perf_counter()
    full_cmd = cmd
    if measure_memory and shutil.which("/usr/bin/time"):
        full_cmd = ["/usr/bin/time", "-v"] + cmd
    proc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    if proc.returncode not in (0, 2):
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}\n{proc.stdout}\n{proc.stderr}")
    peak_rss_mb = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", proc.stderr)
    if m:
        peak_rss_mb = float(m.group(1)) / 1024.0
    return proc.stdout, proc.stderr, t1 - t0, peak_rss_mb


def run_laga(input_path, output_path, schema, target_size, minsup, tau=0.85, iterations=3, seed=42, mode="full", measure_memory=False):
    cmd = [
        str(BIN), "laga",
        "--input", str(input_path),
        "--output", str(output_path),
        "--schema", schema,
        "--target-size", str(target_size),
        "--minsup", str(minsup),
        "--tau-copy", str(tau),
        "--iterations", str(iterations),
        "--support-tolerance", "0.02",
        "--seed", str(seed),
    ]
    if mode == "no_feedback":
        cmd.append("--disable-feedback")
    elif mode == "no_anchors":
        cmd.append("--disable-anchors")
    elif mode == "independent":
        cmd.extend(["--disable-feedback", "--disable-anchors", "--alpha", "100.0"])
    stdout, _, wall, peak_rss_mb = run_cmd(cmd, measure_memory=measure_memory)
    stats = parse_stats(stdout, str(output_path) + ".stats")
    stats["wall_sec"] = wall
    stats["peak_rss_mb"] = peak_rss_mb
    return stats


def make_subset(src, dst, n):
    records = read_records(src, limit=n)
    write_records(dst, records)
    return len(records)


def python_independent_baseline(real_records, target_size, seed):
    rng = random.Random(seed)
    lengths = [len(r) for r in real_records]
    freq = Counter(x for r in real_records for x in set(r))
    items = list(freq)
    weights = [freq[x] for x in items]
    out = []
    for _ in range(target_size):
        L = rng.choice(lengths) if lengths else 1
        chosen = set()
        while len(chosen) < min(L, len(items)):
            chosen.add(rng.choices(items, weights=weights, k=1)[0])
        out.append(tuple(sorted(chosen)))
    return out


def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    dataset = ROOT / "datasets" / "itemsets" / "retail.txt"
    if not dataset.exists():
        dataset = ROOT / "datasets" / "synthetic" / "syn_sparse.txt"
    default_records = int(os.environ.get("LAGA_Q1_RECORDS", "80"))
    base = TMP / f"retail_q1_{default_records}.txt"
    n_real = make_subset(dataset, base, default_records)
    real_records = read_records(base)

    fidelity_rows = []
    support_rows = []
    minsups = [0.01, 0.03, 0.05, 0.10]
    modes = ["full", "independent", "no_feedback", "no_anchors"]
    for minsup in minsups:
        for mode in modes:
            out = OUT / f"exp1_{mode}_s{str(minsup).replace('.', 'p')}.txt"
            st = run_laga(base, out, "transaction", n_real, minsup, tau=1.10, iterations=4, seed=17, mode=mode)
            syn_records = read_records(out)
            pm = pattern_metrics(real_records, syn_records, minsup)
            row = {
                "experiment": "EXP-1",
                "mode": mode,
                "minsup": minsup,
                **{k: v for k, v in pm.items() if not k.endswith("patterns")},
                "copy_rate": st.get("copy_rate", 0.0),
                "nearest_similarity": st.get("avg_nearest_similarity", 0.0),
                "runtime_sec": st.get("runtime_sec", st["wall_sec"]),
            }
            fidelity_rows.append(row)
            if mode == "full" and minsup == 0.03:
                ranked = sorted(pm["real_patterns"], key=lambda p: pm["real_patterns"][p], reverse=True)[:80]
                for rank, pat in enumerate(ranked, 1):
                    support_rows.append({
                        "rank": rank,
                        "pattern": " ".join(pat),
                        "real_support": pm["real_patterns"][pat],
                        "synthetic_support": pm["syn_patterns"].get(pat, 0.0),
                    })

    privacy_rows = []
    for tau in [0.70, 0.80, 0.88, 0.94, 0.98, 0.995]:
        out = OUT / f"exp2_tau_{str(tau).replace('.', 'p')}.txt"
        target_priv = min(n_real, int(os.environ.get("LAGA_Q1_PRIVACY_TARGET", "50")))
        st = run_laga(base, out, "transaction", target_priv, 0.03, tau=tau, iterations=4, seed=23, mode="full")
        syn_records = read_records(out)
        pm = pattern_metrics(real_records, syn_records, 0.03)
        sims = nearest_similarity(syn_records, real_records)
        privacy_rows.append({
            "experiment": "EXP-2",
            "tau_copy": tau,
            "pattern_fidelity": pm["f1"],
            "pattern_recall": pm["recall"],
            "support_dev": pm["support_dev"],
            "privacy_risk": sum(1 for s in sims if s > tau) / len(sims) if sims else 0.0,
            "avg_nearest_similarity": sum(sims) / len(sims) if sims else 0.0,
            "runtime_sec": st.get("runtime_sec", st["wall_sec"]),
        })

    ablation_rows = []
    for mode in ["full", "no_feedback", "no_anchors"]:
        for it in range(1, 7):
            out = OUT / f"exp3_{mode}_it{it}.txt"
            effective_it = it if mode == "full" else 1
            st = run_laga(base, out, "transaction", n_real, 0.03, tau=1.10, iterations=effective_it, seed=31, mode=mode)
            syn_records = read_records(out)
            pm = pattern_metrics(real_records, syn_records, 0.03)
            ablation_rows.append({
                "experiment": "EXP-3",
                "mode": mode,
                "iteration": it,
                "support_loss": pm["support_dev"],
                "entropy_loss": st.get("entropy_loss", 0.0),
                "pattern_recall": pm["recall"],
            })

    scale_rows = []
    for n in [50, 80, 120, 200]:
        subset = TMP / f"scale_{n}.txt"
        actual = make_subset(dataset, subset, n)
        recs = read_records(subset)
        out = OUT / f"exp4_laga_{actual}.txt"
        t0 = time.perf_counter()
        st = run_laga(subset, out, "transaction", actual, 0.03, tau=1.10, iterations=3, seed=41, mode="full", measure_memory=True)
        laga_wall = time.perf_counter() - t0
        size_mb = subset.stat().st_size / (1024 * 1024)
        scale_rows.append({
            "experiment": "EXP-4",
            "method": "LAGA-C99",
            "records": actual,
            "size_mb": size_mb,
            "runtime_sec": st.get("runtime_sec", laga_wall),
            "throughput_mb_s": size_mb / max(st.get("runtime_sec", laga_wall), 1e-9),
            "peak_rss_mb": st.get("peak_rss_mb", 0.0),
        })
        t0 = time.perf_counter()
        py_syn = python_independent_baseline(recs, actual, 41)
        py_wall = time.perf_counter() - t0
        scale_rows.append({
            "experiment": "EXP-4",
            "method": "Python-independent",
            "records": actual,
            "size_mb": size_mb,
            "runtime_sec": py_wall,
            "throughput_mb_s": size_mb / max(py_wall, 1e-9),
            "peak_rss_mb": 0.0,
        })

    write_csv(OUT / "laga_fidelity.csv", fidelity_rows)
    write_csv(OUT / "laga_support_deviation.csv", support_rows)
    write_csv(OUT / "laga_privacy_tradeoff.csv", privacy_rows)
    write_csv(OUT / "laga_ablation.csv", ablation_rows)
    write_csv(OUT / "laga_scalability.csv", scale_rows)
    plot_all(fidelity_rows, support_rows, privacy_rows, ablation_rows, scale_rows)
    print(f"LAGA Q1 experiments complete: {OUT}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_all(fidelity, support, privacy, ablation, scale):
    if not HAS_PLOT:
        return
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.figure(figsize=(7, 4))
    x = [r["rank"] for r in support]
    plt.plot(x, [r["real_support"] for r in support], label="Real data", linewidth=2)
    plt.plot(x, [r["synthetic_support"] for r in support], "--o", markersize=3, label="LAGA synthetic")
    plt.xlabel("Pattern rank by real support")
    plt.ylabel("Support")
    plt.title("Pattern Support Deviation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS / "chart1_pattern_support_deviation.png", dpi=260)
    plt.close()

    methods = ["full", "independent", "no_feedback", "no_anchors"]
    minsups = sorted(set(r["minsup"] for r in fidelity))
    width = 0.18
    xs = list(range(len(minsups)))
    plt.figure(figsize=(8, 4.5))
    for i, mode in enumerate(methods):
        vals = [next((r["f1"] for r in fidelity if r["mode"] == mode and r["minsup"] == s), 0) for s in minsups]
        plt.bar([x + (i - 1.5) * width for x in xs], vals, width=width, label=mode)
    plt.xticks(xs, [f"{s:.0%}" for s in minsups])
    plt.ylim(0, 1.05)
    plt.xlabel("Minimum support")
    plt.ylabel("Pattern F1-score")
    plt.title("Pattern Fidelity Across Minsup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS / "chart2_pattern_f1_recall.png", dpi=260)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    privacy_sorted = sorted(privacy, key=lambda r: r["avg_nearest_similarity"])
    plt.plot([r["avg_nearest_similarity"] for r in privacy_sorted], [r["pattern_fidelity"] for r in privacy_sorted], "-o")
    for r in privacy_sorted:
        plt.annotate(f"tau={r['tau_copy']:.2f}", (r["avg_nearest_similarity"], r["pattern_fidelity"]), fontsize=8)
    plt.xlabel("Privacy risk: avg nearest-record Jaccard")
    plt.ylabel("Pattern fidelity F1")
    plt.title("Utility-Privacy Pareto Frontier")
    plt.tight_layout()
    plt.savefig(CHARTS / "chart3_privacy_utility_pareto.png", dpi=260)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    for mode in ["full", "no_feedback", "no_anchors"]:
        rows = sorted([r for r in ablation if r["mode"] == mode], key=lambda r: r["iteration"])
        plt.plot([r["iteration"] for r in rows], [r["support_loss"] for r in rows], marker="o", label=mode)
    plt.xlabel("Closed-loop iterations")
    plt.ylabel("Support loss")
    plt.title("Ablation Loss Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS / "chart4_ablation_loss_convergence.png", dpi=260)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in sorted(set(r["method"] for r in scale)):
        rows = sorted([r for r in scale if r["method"] == method], key=lambda r: r["records"])
        axes[0].plot([r["records"] for r in rows], [r["runtime_sec"] for r in rows], marker="o", label=method)
        axes[1].plot([r["records"] for r in rows], [r["peak_rss_mb"] for r in rows], marker="s", label=method)
    axes[0].set_xlabel("Records")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Execution Time")
    axes[1].set_xlabel("Records")
    axes[1].set_ylabel("Peak RSS (MB)")
    axes[1].set_title("Memory Footprint")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "chart5_scalability_efficiency.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    run_all()
