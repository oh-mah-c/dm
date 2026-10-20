"""Benchmark of FLCHUSM algorithms (single thread, sequential runs).

Algorithms (build/bin/flchus_miner, standalone, no extra libraries):
  FLCHUSPM      as published (Flb depth pruning; can miss patterns, see notes.md)
  FLCHUSPM-     FLCHUSPM without the Flb test (sound baseline)
  FLCHUS-X      proposed (MLB, MDB, cost-aware width pruning, RedS fixpoint, O(n) selection)
  X-noDESC / X-noCOSTW / X-noFIX / X-noSEL   ablations (one component disabled), default setting only

For each dataset one parameter (ms, mu or mc, percentage form of Truong et al. 2022) is varied while
the two others keep their default value; the values of a sweep go from easy to hard, and once an
algorithm exceeds the time limit the harder values of that sweep are skipped for it.
Each run is repeated REPS times; the output CSV has one row per run.

  python3 docs/core/FLCHUS/bench_flchus.py -> docs/core/FLCHUS/results/bench_flchus.csv
"""
import csv
import os
import re
import subprocess
import time
from pathlib import Path

BIN = "./build/bin/flchus_miner"
DATA = os.environ.get("FLCHUS_DATA", "/home/hutech/self/mri-baselines/flchus_data/qsdb")
OUT = Path(os.environ.get("BENCH_OUT", "docs/core/FLCHUS/results/bench_flchus.csv"))
REPS = int(os.environ.get("REPS", "3"))
TIMEOUT = int(os.environ.get("TIMEOUT", "300"))

# dataset: (file, default (ms, mu, mc), sweeps {param: values easy -> hard})
DATASETS = {
    "Snake": ("snake_qsdb.txt", (60, 5, 5), {"ms": [70, 65, 60, 55, 50], "mu": [7, 6, 5, 4, 3], "mc": [3, 4, 5, 6, 7]}),
    "BMS": ("BMS1_qsdb.txt", (0.03, 0.05, 1), {"ms": [0.05, 0.04, 0.03, 0.025, 0.02], "mu": [0.1, 0.07, 0.05, 0.04, 0.03], "mc": [0.5, 1, 2, 4]}),
    "SIGN": ("SIGN_qsdb.txt", (15, 2, 3), {"ms": [25, 20, 15, 12.5, 10], "mu": [3, 2.5, 2, 1.5, 1], "mc": [2, 3, 4, 5]}),
    "LEVIATHAN": ("LEVIATHAN_qsdb.txt", (2, 1, 3), {"ms": [3, 2.5, 2, 1.5, 1], "mu": [2, 1.5, 1, 0.75, 0.5], "mc": [2, 3, 4, 5]}),
    "BIBLE": ("BIBLE_qsdb.txt", (2, 1, 3), {"ms": [3, 2.5, 2, 1.5, 1], "mu": [2, 1.5, 1, 0.75, 0.5], "mc": [2, 3, 4, 5]}),
    "FIFA": ("FIFA_qsdb.txt", (8, 1, 3), {"ms": [12, 10, 8, 6, 5], "mu": [2, 1.5, 1, 0.75], "mc": [2, 3, 4]}),
}
MAIN = [("FLCHUSPM", "flchuspm", {"FLCHUS_FLB": "1"}),
        ("FLCHUSPM-", "flchuspm", {"FLCHUS_FLB": "0"}),
        ("FLCHUS-X", "flchus_x", {})]
ABLATIONS = [("X-noDESC", "flchus_x", {"FLCHUSX_DESC": "0"}),
             ("X-noCOSTW", "flchus_x", {"FLCHUSX_COSTW": "0"}),
             ("X-noFIX", "flchus_x", {"FLCHUSX_FIX": "0"}),
             ("X-noSEL", "flchus_x", {"FLCHUSX_SEL": "0"})]
FIELDS = ["dataset", "sweep", "ms", "mu", "mc", "algorithm", "rep", "status", "time_s", "wall_s", "max_rss_MB",
          "patterns", "nodes", "extensions"]


def run(algo_id, env_extra, path, ms, mu, mc):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("FLCHUS") or k == "DM_FLCHUS_OUT":
            env.pop(k)
    env.update(env_extra)
    t0 = time.monotonic()
    try:
        p = subprocess.run(["/usr/bin/time", "-f", "RSS_KB=%M", BIN, algo_id, path, str(ms), str(mu), str(mc), "pct"],
                           capture_output=True, text=True, timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"status": f"timeout>{TIMEOUT}s", "wall_s": TIMEOUT}
    g = lambda pat, s=p.stdout: (re.search(pat, s).group(1) if re.search(pat, s) else "")
    return {"status": "ok" if p.returncode == 0 else f"error{p.returncode}", "wall_s": round(time.monotonic() - t0, 3),
            "time_s": g(r"time_s=([\d.]+)"), "patterns": g(r"patterns=(\d+)"), "nodes": g(r"nodes=(\d+)"),
            "extensions": g(r"extensions=(\d+)"),
            "max_rss_MB": round(int(g(r"RSS_KB=(\d+)", p.stderr) or 0) / 1024, 2)}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    def record(ds, sweep, ms, mu, mc, name, algo, env, path):
        for rep in range(1, REPS + 1):
            r = run(algo, env, path, ms, mu, mc)
            row = {"dataset": ds, "sweep": sweep, "ms": ms, "mu": mu, "mc": mc, "algorithm": name, "rep": rep, **r}
            w.writerow(row); f.flush()
            print(row, flush=True)
            if r["status"] != "ok":
                return False
        return True

    for ds, (fname, (ms0, mu0, mc0), sweeps) in DATASETS.items():
        path = os.path.join(DATA, fname)
        for name, algo, env in ABLATIONS:
            record(ds, "ablation", ms0, mu0, mc0, name, algo, env, path)
        for sweep, values in sweeps.items():
            for name, algo, env in MAIN:
                for v in values:
                    ms, mu, mc = (v if sweep == "ms" else ms0), (v if sweep == "mu" else mu0), (v if sweep == "mc" else mc0)
                    if not record(ds, sweep, ms, mu, mc, name, algo, env, path):
                        break                      # harder values of this sweep are skipped for this algorithm
    f.close()


if __name__ == "__main__":
    main()
