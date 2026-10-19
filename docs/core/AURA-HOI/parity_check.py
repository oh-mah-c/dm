"""Set-level agreement of the algorithms on the benchmark datasets.

For every configuration of results/main.csv that at least two algorithms
completed, and for every configuration of results/ablation.csv, each completed
program is run once more with DM_HOI_OUT set. Every output line is one itemset
with its items in increasing order; the lines are sorted and hashed (SHA-256).
A configuration agrees when all programs produce the same hash.

Run from the repository root:
  python3 docs/core/AURA-HOI/parity_check.py   -> docs/core/AURA-HOI/results/parity.csv
"""
import csv
import hashlib
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

BIN = "./build/bin/dm.exe"
RES = Path("docs/core/AURA-HOI/results")
DATA = Path("datasets/itemsets")
FILES = {"mushroom": "mushrooms.txt", "retail": "retail.txt", "accidents": "accidents.txt",
         "kosarak": "kosarak.dat", "T10I4D100K": "T10I4D100K.txt", "chess": "chess.txt", "pumsb": "pumsb.txt"}
PIN_CORE = os.environ.get("PIN_CORE", "3")


def digest(algo, path, alpha, sigma, out, ablation=None):
    base, _, variant = algo.partition(":")
    cmd = ["taskset", "-c", PIN_CORE, BIN, base, str(path), "0", alpha]
    if base == "aura_hoi":
        cmd += [sigma, "0", "sum", "raw"]
    env = dict(os.environ, DM_HOI_OUT=out)
    env.pop("AURA_ABLATION", None)
    flags = {"full": "", "no_residual": "no_tail_cap,no_class_bound"}.get(variant, variant)
    if flags:
        env["AURA_ABLATION"] = flags
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(cmd + ["-t", "1"], capture_output=True, text=True, timeout=900, env=env, check=True)
    lines = sorted(open(out).read().splitlines()) if os.path.exists(out) else []
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), len(lines)


def main():
    groups = defaultdict(list)
    for name in ("main", "ablation"):
        for r in csv.DictReader(open(RES / f"{name}.csv")):
            if r["status"] == "ok":
                groups[(name, r["dataset"], r["alpha"], r["sigma"])].append(r)
    rows = list(csv.DictReader(open(RES / "parity.csv"))) if (RES / "parity.csv").exists() else []
    done = {(r["experiment"], r["dataset"], r["alpha"]) for r in rows}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.txt")
        for (exp, ds, alpha, sigma), rs in groups.items():
            if len(rs) < 2 or (exp, ds, alpha) in done:
                continue
            hashes = {}
            for r in rs:
                h, cnt = digest(r["algorithm"], DATA / FILES[ds], alpha, sigma, out)
                assert cnt == int(r["items"]), (ds, alpha, r["algorithm"], cnt, r["items"])
                hashes[r["algorithm"]] = h
            agree = len(set(hashes.values())) == 1
            rows.append({"experiment": exp, "dataset": ds, "alpha": alpha, "programs": len(hashes),
                         "algorithms": " ".join(sorted(hashes)), "hois": rs[0]["items"],
                         "sha256": next(iter(hashes.values())) if agree else "", "agree": int(agree)})
            print(rows[-1], flush=True)
    with open(RES / "parity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} configurations, {sum(1 - int(r['agree']) for r in rows)} disagreements")


if __name__ == "__main__":
    main()
