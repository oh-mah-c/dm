"""Randomized verification of HEP, DFHOI and AURA-HOI against an exact oracle.

For random small databases, the oracle enumerates every itemset and evaluates
occupancy with rational arithmetic. A case (database, alpha) passes when the
number of HOIs reported by HEP, DFHOI, AURA-HOI and every AURA-HOI ablation
variant equals the oracle count, and the AURA-HOI closed view equals the number
of closed HOIs.

Run from the repository root:
  python3 docs/core/AURA-HOI/oracle_test.py   -> docs/core/AURA-HOI/results/oracle.csv
"""
import csv
import itertools
import math
import os
import random
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

BIN = "./build/bin/dm.exe"
OUT = Path("docs/core/AURA-HOI/results/oracle.csv")
ALPHAS = ("0.1", "0.2", "0.3", "0.4")
ABLATIONS = ("no_tail_cap", "no_class_bound", "no_tail_cap,no_class_bound", "no_arena")
SUITES = [  # (mode, seed, databases)
    ("mixed", 101, 400),
    ("uniform", 202, 400),
]


def count(algo, path, alpha, sigma, view="raw", ablation=None):
    cmd = [BIN, algo, path, "0", alpha, str(sigma)]
    if algo == "aura_hoi":
        cmd += ["0", "sum", view]
    env = dict(os.environ)
    env.pop("AURA_ABLATION", None)
    if ablation:
        env["AURA_ABLATION"] = ablation
    out = subprocess.run(cmd + ["-t", "1"], capture_output=True, text=True, timeout=60, env=env).stdout
    m = re.search(r"Frequent Itemsets\s*:\s*(\d+)", out)
    return int(m.group(1)) if m else -1


def oracle(trans, alpha):
    xi = Fraction(alpha) * len(trans)
    sigma = math.ceil(xi)
    items = sorted({i for t in trans for i in t})
    hois = []
    for k in range(1, len(items) + 1):
        frequent = False
        for X in itertools.combinations(items, k):
            sx = frozenset(X)
            sup = [t for t in trans if sx <= t]
            if len(sup) < sigma:
                continue
            frequent = True
            if k * sum(Fraction(1, len(t)) for t in sup) >= xi:
                hois.append((sx, sup))
        if not frequent:
            break
    closed = sum(1 for X, sup in hois if frozenset.intersection(*sup) == X)
    return len(hois), closed, sigma


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for mode, seed, databases in SUITES:
            rng = random.Random(seed)
            cases = mismatches = 0
            for d in range(databases):
                m = rng.randint(4, 11)
                n = rng.randint(5, 30)
                L = rng.randint(2, m)
                trans = [frozenset(rng.sample(range(1, m + 1), L if mode == "uniform" else rng.randint(1, m)))
                         for _ in range(n)]
                path = os.path.join(tmp, f"{mode}_{d}.txt")
                with open(path, "w") as f:
                    f.writelines(" ".join(map(str, sorted(t))) + "\n" for t in trans)
                for alpha in ALPHAS:
                    raw, closed, sigma = oracle(trans, alpha)
                    got = [count(a, path, alpha, sigma) for a in ("hep", "dfhoi", "aura_hoi")]
                    got += [count("aura_hoi", path, alpha, sigma, ablation=ab) for ab in ABLATIONS]
                    got_closed = count("aura_hoi", path, alpha, sigma, view="closed")
                    cases += 1
                    if any(g != raw for g in got) or got_closed != closed:
                        mismatches += 1
                        print(f"MISMATCH {mode} db={d} alpha={alpha} oracle={raw}/{closed} got={got}/{got_closed}")
            rows.append({"mode": mode, "seed": seed, "databases": databases, "cases": cases,
                         "mismatches": mismatches})
            print(rows[-1], flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "seed", "databases", "cases", "mismatches"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
