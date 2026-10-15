"""Exact verification of mfho_miner against a brute-force oracle.

For random databases, the oracle enumerates every itemset contained in some
transaction, evaluates support and average occupancy with rational
arithmetic, and keeps the maximal frequent high-occupancy itemsets. A case
passes when the miner outputs exactly the same family of itemsets, in every
search mode (full algorithm, closed enumeration without pruning, and each
ablation) and for the two-phase baselines HEP+filter and DFHOI+filter.

With --sum the objective is the one of Deng (2020): maximal itemsets X with
occ(X) = |X| * sum_{T containing X} 1/|T| >= xi, xi a fraction of the number
of transactions.

Run from the repository root:
  python3 docs/core/MFHOI/oracle_test.py         -> docs/core/MFHOI/results/oracle.csv
  python3 docs/core/MFHOI/oracle_test.py --sum   -> docs/core/MFHOI/results/oracle_sum.csv
"""
import csv
import itertools
import math
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

BIN = "./build/bin/mfho_miner"
BIN_HOI = "./build/bin/hoi_filter"   # HEP+filter and DFHOI+filter baselines
OUT = Path("docs/core/MFHOI/results/oracle.csv")
MODES = ("full", "lcm", "eclat", "wubocc", "huopm", "no_tail", "no_hut", "no_sub", "no_pre", "hep", "dfhoi")
SUM = "--sum" in sys.argv
if SUM:
    OUT = Path("docs/core/MFHOI/results/oracle_sum.csv")
    MODES = ("full", "lcm", "eclat", "wubocc", "ubo", "huopm", "no_tail", "no_hut", "no_sub", "no_pre", "hep", "dfhoi")
# --modes m1,m2,...: run only these modes ("+open": all itemsets instead of closed ones)
for a in sys.argv:
    if a.startswith("--modes="):
        MODES = tuple(a.split("=", 1)[1].split(","))
        OUT = OUT.with_name(OUT.stem + "_" + "_".join(m.replace("+", "") for m in MODES)[:60] + ".csv")
XIS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.5", "0.7", "0.9", "1.5")   # fractions of n (>= 1: absolute)
SUPPORTS = ("0.1", "0.2", "0.3")
BETAS = ("0.3", "0.5", "0.7")
SUITES = [  # (name, seed, databases, uniform lengths, items in [lo, hi], noise items)
    ("mixed", 11, 300, False, (4, 10), 0),
    ("uniform", 22, 300, True, (4, 10), 0),
    ("mixed-noise", 33, 300, False, (4, 9), 6),
]


def oracle(trans, alpha, beta):
    n = len(trans)
    sigma = max(1, math.ceil(Fraction(alpha) * n))
    b = Fraction(beta)
    cands = set()
    for t in trans:
        for k in range(1, len(t) + 1):
            cands.update(frozenset(x) for x in itertools.combinations(sorted(t), k))
    fho = []
    for X in cands:
        sup = [t for t in trans if X <= t]
        if len(sup) >= sigma and Fraction(len(X), len(sup)) * sum(Fraction(1, len(t)) for t in sup) >= b:
            fho.append(X)
    return {X for X in fho if not any(X < Y for Y in fho)}


def oracle_sum(trans, xi):
    xi = Fraction(xi) * len(trans) if Fraction(xi) < 1 else Fraction(xi)
    cands = set()
    for t in trans:
        for k in range(1, len(t) + 1):
            cands.update(frozenset(x) for x in itertools.combinations(sorted(t), k))
    hoi = [X for X in cands
           if len(X) * sum(Fraction(1, len(t)) for t in trans if X <= t) >= xi]
    return {X for X in hoi if not any(X < Y for Y in hoi)}


def mine(path, alpha, beta, mode, out):
    base, _, extra = mode.partition("+")
    cmd = [BIN_HOI if base in ("hep", "dfhoi") else BIN, path, alpha, beta, "--mode", base, "--out", out]
    if extra == "open":
        cmd.append("--open")
    if SUM:
        cmd.append("--sum")
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    with open(out) as f:
        return {frozenset(map(int, line.split())) for line in f if line.strip()}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, seed, databases, uniform, (lo, hi), noise in SUITES:
            rng = random.Random(seed)
            cases = mismatches = 0
            for d in range(databases):
                m = rng.randint(lo, hi)
                n = rng.randint(4, 24)
                L = rng.randint(2, m)
                trans = []
                for _ in range(n):
                    k = L if uniform else rng.randint(1, m)
                    t = set(rng.sample(range(1, m + 1), k))
                    # items that occur once: they lengthen transactions without being frequent
                    t |= {100 + rng.randint(0, 1000) for _ in range(rng.randint(0, noise))}
                    trans.append(frozenset(t))
                path = os.path.join(tmp, "db.txt")
                with open(path, "w") as f:
                    f.writelines(" ".join(map(str, sorted(t))) + "\n" for t in trans)
                grid = [(x, "0") for x in XIS] if SUM else [(a, b) for a in SUPPORTS for b in BETAS]
                for alpha, beta in grid:
                    if True:
                        exp = oracle_sum(trans, alpha) if SUM else oracle(trans, alpha, beta)
                        cases += 1
                        bad = [mo for mo in MODES if mine(path, alpha, beta, mo, os.path.join(tmp, "o.txt")) != exp]
                        if bad:
                            mismatches += 1
                            print(f"MISMATCH {name} db={d} alpha={alpha} beta={beta} modes={bad} "
                                  f"db={[sorted(t) for t in trans]}", flush=True)
            rows.append({"suite": name, "seed": seed, "databases": databases, "cases": cases,
                         "modes": len(MODES), "mismatches": mismatches})
            print(rows[-1], flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
