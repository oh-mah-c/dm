"""Exact verification of HEP, DFHOI and AURA-HOI against a brute-force oracle.

For random small databases, the oracle enumerates every itemset contained in a
transaction and evaluates occ(X) = |X| * sum_{T containing X} 1/|T| with rational
arithmetic. A case (database, alpha) passes when the FAMILY of itemsets written
by each program (environment variable DM_HOI_OUT) equals the oracle's family of
HOIs, for HEP, DFHOI, AURA-HOI and each AURA-HOI ablation variant.

Thresholds per database:
  * fixed values of alpha;
  * near-tie values: alpha is the occupancy of a random itemset divided by n,
    rounded to 12 significant digits, so that xi lies within about 1e-12 * n of
    an occupancy, on either side;
  * exact ties (suite "ties"): lengths in {1, 2, 4, 5, 8, 10} and n in
    {4, 5, 8, 10, 16, 20, 25}, so that occ(X) / n is a terminating decimal and
    alpha = occ(X) / n gives occ(X) = xi exactly.

Run from the repository root:
  python3 docs/core/AURA-HOI/oracle_test.py   -> docs/core/AURA-HOI/results/oracle.csv
"""
import csv
import itertools
import math
import os
import random
import subprocess
import tempfile
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

BIN = "./build/bin/dm.exe"
OUT = Path("docs/core/AURA-HOI/results/oracle.csv")
FIXED = ("0.05", "0.1", "0.2", "0.3", "0.4", "0.5")
ABLATIONS = ("no_tail_cap", "no_class_bound", "no_tail_cap,no_class_bound", "no_arena")
SUITES = [  # (name, seed, databases)
    ("mixed", 101, 300),
    ("uniform", 202, 300),
    ("ties", 303, 300),
]


def run(algo, path, alpha, sigma, out, ablation=None):
    cmd = [BIN, algo, path, "0", alpha]
    if algo == "aura_hoi":
        cmd += [str(sigma), "0", "sum", "raw"]
    env = dict(os.environ, DM_HOI_OUT=out)
    env.pop("AURA_ABLATION", None)
    if ablation:
        env["AURA_ABLATION"] = ablation
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(cmd + ["-t", "1"], capture_output=True, text=True, timeout=120, env=env, check=True)
    if not os.path.exists(out):
        return set()
    with open(out) as f:
        return {frozenset(map(int, line.split())) for line in f if line.strip()}


def oracle(trans, alpha):
    xi = Fraction(alpha) * len(trans)
    cands = set()
    for t in trans:
        for k in range(1, len(t) + 1):
            cands.update(frozenset(c) for c in itertools.combinations(sorted(t), k))
    return {X for X in cands if len(X) * sum(Fraction(1, len(t)) for t in trans if X <= t) >= xi}


def occ(trans, X):
    return len(X) * sum(Fraction(1, len(t)) for t in trans if X <= t)


def make_db(rng, suite):
    if suite == "ties":
        n = rng.choice([4, 5, 8, 10, 16, 20, 25])
        m = rng.randint(4, 10)
        trans = []
        for _ in range(n):
            L = rng.choice([L for L in (1, 2, 4, 5, 8, 10) if L <= m])
            trans.append(frozenset(rng.sample(range(1, m + 1), L)))
        return trans
    m = rng.randint(4, 11)
    n = rng.randint(5, 30)
    L = rng.randint(2, m)
    return [frozenset(rng.sample(range(1, m + 1), L if suite == "uniform" else rng.randint(1, m)))
            for _ in range(n)]


def alphas_for(rng, trans, suite):
    n = len(trans)
    out = list(FIXED)
    items = sorted({i for t in trans for i in t})
    for _ in range(3):
        t = rng.choice(trans)
        X = frozenset(rng.sample(sorted(t), rng.randint(1, len(t))))
        a = occ(trans, X) / n
        if suite == "ties":
            dec = Decimal(a.numerator) / Decimal(a.denominator)   # terminating decimal
            out.append(format(dec.normalize(), "f"))
        else:
            out.append(f"{float(a):.12g}")
    return [a for a in out if 0 < Fraction(a) < 1]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "db.txt")
        out = os.path.join(tmp, "out.txt")
        for suite, seed, databases in SUITES:
            rng = random.Random(seed)
            cases = mismatches = ties = 0
            for d in range(databases):
                trans = make_db(rng, suite)
                with open(path, "w") as f:
                    f.writelines(" ".join(map(str, sorted(t))) + "\n" for t in trans)
                for alpha in alphas_for(rng, trans, suite):
                    exp = oracle(trans, alpha)
                    xi = Fraction(alpha) * len(trans)
                    sigma = max(1, math.ceil(xi))
                    if any(occ(trans, X) == xi for X in exp):
                        ties += 1
                    got = {"hep": run("hep", path, alpha, sigma, out),
                           "dfhoi": run("dfhoi", path, alpha, sigma, out),
                           "aura_hoi": run("aura_hoi", path, alpha, sigma, out)}
                    for abl in ABLATIONS:
                        got["aura_hoi:" + abl] = run("aura_hoi", path, alpha, sigma, out, abl)
                    cases += 1
                    bad = [k for k, v in got.items() if v != exp]
                    if bad:
                        mismatches += 1
                        print(f"MISMATCH {suite} db={d} alpha={alpha} {bad} "
                              f"db={[sorted(t) for t in trans]}", flush=True)
            rows.append({"suite": suite, "seed": seed, "databases": databases, "cases": cases,
                         "cases_with_exact_tie": ties, "programs": 3 + len(ABLATIONS), "mismatches": mismatches})
            print(rows[-1], flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
