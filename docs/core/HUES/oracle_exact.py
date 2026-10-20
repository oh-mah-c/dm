"""Brute-force oracle test of hue_exact.c (reference miner) on random small sequences.

  gcc -O2 -o <bin> docs/core/HUES/hue_exact.c
  HUE_EXACT=<bin> python3 docs/core/HUES/oracle_exact.py -> results/oracle_exact.csv
"""
import csv
import importlib.util
import os
import random
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location("o", os.path.join(os.path.dirname(__file__), "oracle_huespan.py"))
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
BIN = os.environ["HUE_EXACT"]
SEQUENCES = int(os.environ.get("SEQUENCES", "300"))
MAXDURS = [int(x) for x in os.environ.get("MAXDURS", "1,2,3,4,5,6").split(",")]
SEED = 41


def parse(path):
    res = {}
    for line in open(path):
        body, util = line.split("#UTIL:")
        sets, cur = [], []
        for tok in body.split():
            if tok == "-1":
                sets.append(frozenset(cur)); cur = []
            else:
                cur.append(int(tok))
        res[tuple(sets)] = int(util)
    return res


def main():
    rng = random.Random(SEED)
    cases = bad = missing = extra = 0
    with tempfile.TemporaryDirectory() as tmp:
        path, out = os.path.join(tmp, "s.txt"), os.path.join(tmp, "o.txt")
        for s in range(SEQUENCES):
            seq = o.make_sequence(rng)
            open(path, "w").write(o.to_spmf(seq))
            ratio, md = rng.choice([0.05, 0.1, 0.2, 0.3, 0.5]), rng.choice(MAXDURS)
            subprocess.run([BIN, path, str(ratio), str(md), out] + os.environ.get("EXTRA_ARGS", "").split(), check=True, capture_output=True)
            got, exp = parse(out), o.oracle(seq, ratio, md, "I1")
            cases += 1
            if got != exp:
                bad += 1
                missing += len(set(exp) - set(got))
                extra += len([e for e in got if exp.get(e) != got[e]])
                if bad <= 3:
                    print("MISMATCH", ratio, md, seq, sorted(set(exp) ^ set(got), key=len)[:3], flush=True)
            if (s + 1) % 50 == 0:
                print(f"{s + 1} cases, mismatching {bad}", flush=True)
    row = {"cases": cases, "seed": SEED, "maxDurs": " ".join(map(str, MAXDURS)), "mismatching_cases": bad,
           "missing": missing, "extra_or_wrong": extra}
    os.makedirs("docs/core/HUES/results", exist_ok=True)
    with open(f"docs/core/HUES/results/oracle_{os.environ.get('TAG', 'exact')}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
