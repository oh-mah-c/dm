"""Compare the MHOI sets (not only their numbers) returned by the three methods
on every benchmark configuration that at least two of them completed.

  python3 docs/core/MFHOI/parity_sets.py -> docs/core/MFHOI/results/parity_sets.csv
"""
import csv
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

RUNS = Path("docs/core/MFHOI/results/runs.csv")
OUT = Path("docs/core/MFHOI/results/parity_sets.csv")
FILES = {"mushroom": "mushrooms.txt", "retail": "retail.txt", "accidents": "accidents.txt",
         "kosarak": "kosarak.dat", "T10I4D100K": "T10I4D100K.txt"}


def mine(method, path, xi, out):
    binary = "./build/bin/hoi_filter" if method in ("hep", "dfhoi") else "./build/bin/mfho_miner"
    mode, extra = ("full", ["--open"]) if method == "mhoi" else (method, [])
    subprocess.run([binary, str(path), xi, "0", "--mode", mode, "--sum", "--out", out] + extra, check=True,
                   capture_output=True, timeout=1200)
    sets = sorted(" ".join(sorted(line.split(), key=int)) for line in open(out) if line.strip())
    return len(sets), hashlib.sha256("\n".join(sets).encode()).hexdigest()


def main():
    ok = {}
    for r in csv.DictReader(open(RUNS)):
        if r["experiment"] == "xi" and r["status"] == "ok" and r["method"] in ("mhoi", "hep", "dfhoi"):
            ok.setdefault((r["dataset"], r["xi_rel"]), []).append(r["method"])
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for (d, x), methods in sorted(ok.items()):
            if len(methods) < 2:
                continue
            res = {m: mine(m, Path("datasets/itemsets") / FILES[d], x, os.path.join(tmp, m)) for m in methods}
            same = len({v for v in res.values()}) == 1
            rows.append({"dataset": d, "xi_rel": x, "methods": " ".join(sorted(methods)),
                         "mhoi": next(iter(res.values()))[0], "identical": int(same)})
            print(rows[-1], flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("configurations", len(rows), "identical", sum(r["identical"] for r in rows))


if __name__ == "__main__":
    main()
