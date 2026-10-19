"""Single-threaded benchmark suite for the AURA-HOI paper.

Experiments (run from the repository root):
  python3 docs/core/AURA-HOI/run_benchmarks.py main         -> results/main.csv
  python3 docs/core/AURA-HOI/run_benchmarks.py ablation     -> results/ablation.csv
  python3 docs/core/AURA-HOI/run_benchmarks.py scalability  -> results/scalability.csv

Datasets and thresholds: those of Deng (2020), Table 3, the same as in the MFHOI
experiments: mushroom 12-4%, retail 0.05-0.01%, accidents 30-10%, kosarak 1-0.2%,
T10I4D100K 0.1-0.02% of the number of transactions (five values each), and the
additional dense datasets of the DFHOI evaluation (Nguyen et al. 2023, Table 8):
chess 48-40% and pumsb 86-78%.

Protocol (identical to docs/core/MFHOI/run_benchmarks.py)
  * every run uses one thread (-t 1) and is pinned with taskset to core PIN_CORE;
  * sigma = ceil(alpha * n) is computed exactly with rational arithmetic;
  * runtime is the mining phase reported by the binary (data loading excluded);
  * memory is the peak RSS increase during the mining phase ("Algo Peak Extra");
  * wall-clock limit of 600 s per run (AURA_TIMEOUT_S);
  * an address-space limit protects the machine (AURA_MEM_LIMIT_GB, default 20; 0 disables it);
  * repetitions: 5 if the first run takes < 1 s, 3 if < 60 s, otherwise 1; the median is reported;
  * thresholds are processed from the largest to the smallest; once an algorithm fails on a
    dataset, smaller thresholds are recorded as "skipped": every pruning rule of the three
    algorithms compares a bound with xi, so lowering alpha can only enlarge the search;
  * the runs are resumable: rows already present in the output CSV are kept and not re-run,
    so an interrupted experiment continues where it stopped (delete the CSV to start afresh).
"""
import csv
import math
import os
import re
import resource
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

BIN = "./build/bin/dm.exe"
OUT = Path("docs/core/AURA-HOI/results")
TIMEOUT_S = float(os.environ.get("AURA_TIMEOUT_S", "600"))
PIN_CORE = os.environ.get("AURA_PIN_CORE", "3")
MEM_LIMIT_GB = float(os.environ.get("AURA_MEM_LIMIT_GB", "20"))
ALGOS = ("hep", "dfhoi", "aura_hoi")

DATA = Path("datasets/itemsets")
DATASETS = {
    # name: (file, alphas); the thresholds of Deng (2020), Table 3
    "mushroom":   ("mushrooms.txt",  [0.12, 0.10, 0.08, 0.06, 0.04]),
    "retail":     ("retail.txt",     [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]),
    "accidents":  ("accidents.txt",  [0.30, 0.25, 0.20, 0.15, 0.10]),
    "kosarak":    ("kosarak.dat",    [0.01, 0.008, 0.006, 0.004, 0.002]),
    "T10I4D100K": ("T10I4D100K.txt", [0.001, 0.0008, 0.0006, 0.0004, 0.0002]),
    # additional dense datasets; the thresholds of Nguyen et al. (2023), Table 8
    "chess":      ("chess.txt",      [0.48, 0.46, 0.44, 0.42, 0.40]),
    "pumsb":      ("pumsb.txt",      [0.86, 0.84, 0.82, 0.80, 0.78]),
}

RE = {
    "time": re.compile(r"Algorithm Core\s*:\s*([0-9.]+)\s*ms"),
    "mem": re.compile(r"Algo Peak Extra\s*:\s*([0-9.]+)\s*MB"),
    "items": re.compile(r"Frequent Itemsets\s*:\s*(\d+)"),
    "nodes": re.compile(r"visited_nodes=(\d+)"),
}


def n_transactions(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def limit_memory():
    if MEM_LIMIT_GB <= 0:
        return
    lim = int(MEM_LIMIT_GB * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def run_once(algo, path, alpha, sigma, env_extra=None):
    cmd = ["taskset", "-c", PIN_CORE, BIN, algo, str(path), "0", str(alpha)]
    if algo == "aura_hoi":
        cmd += [str(sigma), "0", "sum", "raw"]
    cmd += ["-t", "1"]
    env = dict(os.environ)
    env.pop("AURA_ABLATION", None)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
                           preexec_fn=limit_memory, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    out = p.stdout
    m = {k: rx.search(out) for k, rx in RE.items()}
    if p.returncode != 0 or not m["time"] or not m["items"]:
        # The baselines do not check malloc results, so exhausting the
        # address-space limit shows up as a crash (negative return code).
        return {"status": "memory" if p.returncode < 0 else "failed", "returncode": p.returncode}
    return {"status": "ok", "time": float(m["time"].group(1)) / 1000.0,
            "mem": float(m["mem"].group(1)) if m["mem"] else float("nan"),
            "items": int(m["items"].group(1)), "nodes": int(m["nodes"].group(1)) if m["nodes"] else -1}


def measure(algo, path, alpha, sigma, env_extra=None):
    first = run_once(algo, path, alpha, sigma, env_extra)
    if first["status"] != "ok":
        return dict(first, reps=1)
    reps = 5 if first["time"] < 1 else (3 if first["time"] < 60 else 1)
    runs = [first] + [run_once(algo, path, alpha, sigma, env_extra) for _ in range(reps - 1)]
    if any(r["status"] != "ok" for r in runs):
        return {"status": "unstable", "reps": reps}
    if len({(r["items"], r["nodes"]) for r in runs}) != 1:
        return {"status": "nondeterministic", "reps": reps}
    t = [r["time"] for r in runs]
    return {"status": "ok", "reps": reps, "time": statistics.median(t), "time_min": min(t),
            "time_max": max(t), "mem": max(r["mem"] for r in runs),
            "items": runs[0]["items"], "nodes": runs[0]["nodes"]}


FIELDS = ["experiment", "dataset", "variant", "fraction", "n", "alpha", "sigma", "algorithm", "status",
          "returncode", "reps", "time", "time_min", "time_max", "mem", "items", "nodes"]


DONE = {}


def key(row):
    return (row["dataset"], str(row["variant"]), str(float(row["fraction"])), str(float(row["alpha"])),
            row["algorithm"])


def writer(name):
    """Open the output CSV in append mode and remember the rows it already holds."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    DONE.clear()
    if path.exists() and path.stat().st_size > 0:
        for r in csv.DictReader(open(path)):
            DONE[key(r)] = r
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if not DONE:
        w.writeheader()
    return f, w


def emit(f, w, row):
    w.writerow({k: row.get(k, "") for k in FIELDS})
    f.flush()
    print(" ".join(f"{k}={row.get(k, '')}" for k in ("dataset", "fraction", "variant", "alpha", "algorithm",
                                                       "status", "reps", "time", "items", "nodes", "mem")),
          flush=True)


def sweep(f, w, experiment, ds, path, alphas, algos, variant="", fraction=1.0, env_for=None):
    n = n_transactions(path)
    failed = set()
    for alpha in sorted(alphas, reverse=True):
        sigma = math.ceil(Fraction(str(alpha)) * n)
        for algo in algos:
            row = {"experiment": experiment, "dataset": ds, "variant": variant, "fraction": fraction,
                   "n": n, "alpha": alpha, "sigma": sigma, "algorithm": algo}
            if key(row) in DONE:
                if DONE[key(row)]["status"] != "ok":
                    failed.add(algo)
                continue
            if algo in failed:
                row.update(status="skipped")
            else:
                row.update(measure(algo.split(":")[0], path, alpha, sigma, env_for(algo) if env_for else None))
                if row["status"] != "ok":
                    failed.add(algo)
            emit(f, w, row)


def main_experiment():
    f, w = writer("main")
    for ds, (fn, alphas) in DATASETS.items():
        sweep(f, w, "main", ds, DATA / fn, alphas, ALGOS)


ABLATION = {
    "full": "",
    "no_tail_cap": "no_tail_cap",
    "no_class_bound": "no_class_bound",
    "no_residual": "no_tail_cap,no_class_bound",
    "no_arena": "no_arena",
}
# middle threshold of each dataset
ABLATION_CONFIGS = [(ds, alphas[2]) for ds, (fn, alphas) in DATASETS.items()]


def ablation_experiment():
    f, w = writer("ablation")
    for ds, alpha in ABLATION_CONFIGS:
        fn = DATASETS[ds][0]
        for variant, flags in ABLATION.items():
            sweep(f, w, "ablation", ds, DATA / fn, [alpha], [f"aura_hoi:{variant}"], variant=variant,
                  env_for=lambda a, fl=flags: {"AURA_ABLATION": fl} if fl else None)


SCALABILITY = {"kosarak": DATASETS["kosarak"][1][2], "T10I4D100K": DATASETS["T10I4D100K"][1][2]}
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def scalability_experiment():
    f, w = writer("scalability")
    tmp = OUT / "prefixes"
    tmp.mkdir(parents=True, exist_ok=True)
    for ds, alpha in SCALABILITY.items():
        src = DATA / DATASETS[ds][0]
        lines = [ln for ln in open(src) if ln.strip()]
        for frac in FRACTIONS:
            path = tmp / f"{ds}_{int(frac * 100)}.txt"
            with open(path, "w") as g:
                g.writelines(lines[: int(round(frac * len(lines)))])
            sweep(f, w, "scalability", ds, path, [alpha], ALGOS, fraction=frac)


if __name__ == "__main__":
    {"main": main_experiment, "ablation": ablation_experiment,
     "scalability": scalability_experiment}[sys.argv[1]]()
