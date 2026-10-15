"""Benchmarks for maximal high occupancy itemset (MHOI) mining.

Problem (Deng 2020): occ(X) = |X| * sum_{T containing X} 1/|T|; X is a high
occupancy itemset (HOI) if occ(X) >= xi; an MHOI is an HOI with no HOI proper
superset. As in Deng (2020), xi is given as a fraction of the number of
transactions, and the datasets and threshold ranges are those of that paper.

Run from the repository root:
  python3 docs/core/MFHOI/run_benchmarks.py [xi] [ablation]
      -> docs/core/MFHOI/results/runs.csv, log on stdout

Methods (the two published HOI miners, each followed by naive maximality
filtering, against the proposed algorithm)
  mhoi   : the proposed algorithm (mfho_miner --sum --open, same algorithm as
           src/algorithms/mfhoi.c)
  hep    : HEP+filter, HEP (Deng 2020) followed by the maximality filter
  dfhoi  : DFHOI+filter, DFHOI (Nguyen et al. 2023) followed by the same filter

Protocol
  * one thread, pinned with taskset to CPU core PIN_CORE;
  * runtime = mining + final filtering (reading the file and building the
    initial database representation excluded), as reported by the binary;
  * wall-clock limit TIMEOUT_S and address-space limit MEM_LIMIT_GB per run;
  * repetitions: 5 if the first run takes < 1 s, 3 if < 60 s, otherwise 1;
    the median is reported, and the min and max are kept;
  * thresholds are processed from the largest to the smallest; once a method
    fails, the smaller thresholds are recorded as skipped (a smaller xi only
    enlarges the search of every method);
  * resumable: rows already in the output CSV are not run again.
"""
import csv
import re
import resource
import statistics
import subprocess
import sys
from pathlib import Path
import os

BIN = "./build/bin/mfho_miner"
BIN_HOI = "./build/bin/hoi_filter"
OUT = Path("docs/core/MFHOI/results/runs.csv")
DATA = Path("datasets/itemsets")
PIN_CORE = os.environ.get("MFHO_PIN_CORE", "3")
TIMEOUT_S = float(os.environ.get("MFHO_TIMEOUT_S", "600"))
MEM_LIMIT_GB = float(os.environ.get("MFHO_MEM_LIMIT_GB", "20"))

# "mhoi" is the proposed algorithm (mfho_miner --mode full --open: all itemsets, rules P1-P4);
# "full" is the earlier closed-itemset version, kept for the ablation
METHODS = ["mhoi", "hep", "dfhoi"]
MODE_ARGS = {"mhoi": ["full", "--open"], "no_pre_o": ["no_pre", "--open"], "wubocc_o": ["wubocc", "--open"],
             "ubo_o": ["ubo", "--open"], "no_hut_o": ["no_hut", "--open"], "no_sub_o": ["no_sub", "--open"]}
# ablation variants of mfho_miner, at the middle threshold of each dataset:
#   no_pre : bound B in rule P1, rule P4 off
#   wubocc : per-transaction bound W in rule P1, rule P4 off
#   ubo    : bound UBO of Deng (2020) in rule P1, rule P4 off
#   no_hut, no_sub : rule P2, resp. P3, off
#   eclat  : all itemsets instead of closed itemsets (same rules)
ABLATION = ["mhoi", "no_pre_o", "wubocc_o", "ubo_o", "no_hut_o", "no_sub_o", "full"]

# dataset: (file, xi grid as fractions of n, from largest to smallest).
# Ranges follow Deng (2020), Table 3: mushroom 4-12%, retail 0.01-0.05%,
# accidents 10-30%, kosarak 0.2-1%, T10I4D100K 0.02-0.1%.
DATASETS = {
    "mushroom":   ("mushrooms.txt",  [0.12, 0.10, 0.08, 0.06, 0.04]),
    "retail":     ("retail.txt",     [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]),
    "accidents":  ("accidents.txt",  [0.30, 0.25, 0.20, 0.15, 0.10]),
    "kosarak":    ("kosarak.dat",    [0.01, 0.008, 0.006, 0.004, 0.002]),
    "T10I4D100K": ("T10I4D100K.txt", [0.001, 0.0008, 0.0006, 0.0004, 0.0002]),
}

FIELDS = ["experiment", "dataset", "xi_rel", "xi", "method", "status", "reps",
          "time", "time_min", "time_max", "nodes", "closure_evals", "mhoi", "candidates",
          "hoi", "peak_rss_kb", "rss_after_load_kb", "time_load"]

RX = {k: re.compile(p, re.M) for k, p in {
    "xi": r"xi=([0-9.]+)", "mhoi": r"^mfhoi=(\d+)", "candidates": r"candidates=(\d+)",
    "nodes": r"^nodes=(\d+)", "closure_evals": r"closure_evals=(\d+)",
    "hoi": r"\b(?:fho_nodes|hoi)=(\d+)",
    "time_mine": r"time_mine=([0-9.]+)", "time_filter": r"time_filter=([0-9.]+)",
    "time_load": r"time_load=([0-9.]+)", "peak_rss_kb": r"peak_rss_kb=(\d+)",
    "rss_after_load_kb": r"rss_after_load_kb=(\d+)"}.items()}


def limit_memory():
    if MEM_LIMIT_GB > 0:
        lim = int(MEM_LIMIT_GB * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def run_once(path, xi, method, outfile=None):
    binary = BIN_HOI if method in ("hep", "dfhoi") else BIN
    mode, *extra = MODE_ARGS.get(method, [method])
    cmd = ["taskset", "-c", PIN_CORE, binary, str(path), str(xi), "0", "--mode", mode, "--sum"] + extra
    if outfile:
        cmd += ["--out", outfile]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, preexec_fn=limit_memory)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if p.returncode != 0:
        return {"status": "memory" if p.returncode < 0 else "failed"}
    out = {}
    for k, rx in RX.items():
        m = rx.search(p.stdout)
        out[k] = m.group(1) if m else ""
    out["time"] = float(out["time_mine"]) + float(out["time_filter"])
    out["status"] = "ok"
    return out


def measure(path, xi, method):
    first = run_once(path, xi, method)
    if first["status"] != "ok":
        return dict(first, reps=1)
    reps = 5 if first["time"] < 1 else (3 if first["time"] < 60 else 1)
    runs = [first] + [run_once(path, xi, method) for _ in range(reps - 1)]
    if any(r["status"] != "ok" for r in runs):
        return {"status": "unstable", "reps": reps}
    if len({(r["mhoi"], r["nodes"]) for r in runs}) != 1:
        return {"status": "nondeterministic", "reps": reps}
    t = [r["time"] for r in runs]
    row = dict(runs[0])
    row.update(status="ok", reps=reps, time=statistics.median(t), time_min=min(t), time_max=max(t),
               peak_rss_kb=max(int(r["peak_rss_kb"]) for r in runs))
    return row


def key(r):
    return (r["experiment"], r["dataset"], str(float(r["xi_rel"])), r["method"])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if OUT.exists() and OUT.stat().st_size:
        for r in csv.DictReader(open(OUT)):
            done[key(r)] = r
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if not done:
        w.writeheader()

    def sweep(experiment, ds, grid, methods):
        path = DATA / DATASETS[ds][0]
        failed = set()
        for xi in grid:
            for method in methods:
                row = {"experiment": experiment, "dataset": ds, "xi_rel": xi, "method": method}
                if key(row) in done:
                    if done[key(row)]["status"] != "ok":
                        failed.add(method)
                    continue
                if method in failed:
                    row["status"] = "skipped"
                else:
                    row.update(measure(path, xi, method))
                    if row["status"] != "ok":
                        failed.add(method)
                w.writerow(row)
                f.flush()
                print(experiment, ds, xi, method, row.get("status"), row.get("time", ""),
                      row.get("nodes", ""), row.get("mhoi", ""), flush=True)

    only = sys.argv[1:] or ["xi"]
    for ds, (fn, grid) in DATASETS.items():
        if "xi" in only:
            sweep("xi", ds, grid, METHODS)
        if "ablation" in only:
            sweep("ablation", ds, [grid[2]], ABLATION)


if __name__ == "__main__":
    main()
