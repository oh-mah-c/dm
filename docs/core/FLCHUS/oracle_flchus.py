"""Brute-force oracle for FLCHUSM (Truong et al., Applied Intelligence 52 (2022) 7136-7157,
Definitions 1-4 and the problem definition of Section 3), compared with dm.exe flchuspm.

  - a pattern is a list of itemsets; it occurs in a q-sequence if its itemsets are subsets of
    itemsets of the q-sequence at strictly increasing positions;
  - u(alpha, Psi) = max over occurrences of the sum of the utilities of the matched q-items,
    c(alpha, Psi) = max over occurrences of the sum of their costs (taken independently);
  - supp = number of q-sequences containing alpha, au = sum u / supp, ac = sum c / supp;
  - FLCHUS = {alpha | supp >= mins, au >= minu, ac <= maxc}.
All patterns with supp >= mins are enumerated (support is anti-monotonic). Random small QSDBs.

Run from the repository root:
  python3 docs/core/FLCHUS/oracle_flchus.py -> docs/core/FLCHUS/results/oracle_flchus.csv
"""
import csv
import math
import os
import random
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

BIN = "./build/bin/dm.exe"
OUT = Path(os.environ.get("ORACLE_OUT", "docs/core/FLCHUS/results/oracle_flchus.csv"))
CASES = int(os.environ.get("CASES", "500"))
SEED = int(os.environ.get("SEED", "7"))
EXTRA_ENV = dict(e.split("=", 1) for e in os.environ.get("FLCHUS_ENV", "").split() if "=" in e)


LONG = os.environ.get("LONG") == "1"   # longer sequences over fewer items (many repetitions)


def make_db(rng):
    n_items = rng.randint(2, 3) if LONG else rng.randint(2, 5)
    db = []
    for _ in range(rng.randint(2, 7)):
        seq = []
        for _ in range(rng.randint(3, 8) if LONG else rng.randint(1, 5)):
            items = rng.sample(range(1, n_items + 1), rng.randint(1, min(3, n_items)))
            seq.append({i: (rng.randint(1, 20), rng.randint(1, 50)) for i in items})
        db.append(seq)
    return db


def to_file(db):
    lines = []
    for seq in db:
        toks = []
        for its in seq:
            toks += [f"{i}:{c}:{u}" for i, (c, u) in sorted(its.items())] + ["-1"]
        lines.append(" ".join(toks + ["-2"]))
    return "\n".join(lines) + "\n"


def measures(seq, pat):
    """(contained, max utility, max cost) of pattern pat in q-sequence seq"""
    n, k = len(seq), len(pat)

    @lru_cache(maxsize=None)
    def best(i, j, which):
        # best value matching pattern itemsets i.. using sequence itemsets j..; -inf if impossible
        if i == k:
            return 0
        res = -math.inf
        for p in range(j, n):
            if pat[i] <= seq[p].keys():
                v = best(i + 1, p + 1, which)
                if v > -math.inf:
                    res = max(res, v + sum(seq[p][x][which] for x in pat[i]))
        return res
    u = best(0, 0, 1)
    if u == -math.inf:
        return False, 0, 0
    return True, u, best(0, 0, 0)


def oracle(db, mins, minu, maxc):
    items = sorted({i for s in db for its in s for i in its})
    res = {}

    def visit(pat):
        rows = [measures(s, pat) for s in db]
        sup = sum(1 for r in rows if r[0])
        if sup < mins:
            return
        au = sum(r[1] for r in rows if r[0]) / sup
        ac = sum(r[2] for r in rows if r[0]) / sup
        if au >= minu - 1e-9 and ac <= maxc + 1e-9:
            res[pat] = (sup, au, ac)
        last = pat[-1]
        for y in items:
            if y > max(last):
                visit(pat[:-1] + (last | frozenset([y]),))
            visit(pat + (frozenset([y]),))

    for y in items:
        visit((frozenset([y]),))
    return res


def run_port(path, mins, minu, maxc, out):
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ, DM_FLCHUS_OUT=out, **EXTRA_ENV)
    subprocess.run([BIN, os.environ.get("ALGO", "flchuspm"), path, "0", str(mins), repr(minu), repr(maxc)], env=env,
                   capture_output=True, text=True, check=True, timeout=120)
    res = {}
    for line in open(out):
        body, rest = line.split("#SUP:")
        sets, cur = [], []
        for tok in body.split():
            if tok == "-1":
                sets.append(frozenset(cur)); cur = []
            else:
                cur.append(int(tok))
        sup = int(rest.split()[0])
        au = float(rest.split("#AU:")[1].split()[0])
        ac = float(rest.split("#AC:")[1].split()[0])
        res[tuple(sets)] = (sup, au, ac)
    return res


def fmt(p):
    return "->".join("(" + ",".join(map(str, sorted(s))) + ")" for s in p)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    bad = missing = extra = wrong = 0
    total_patterns = 0
    shown = 0
    with tempfile.TemporaryDirectory() as tmp:
        path, out = os.path.join(tmp, "db.txt"), os.path.join(tmp, "out.txt")
        for case in range(CASES):
            db = make_db(rng)
            open(path, "w").write(to_file(db))
            mins = rng.randint(1, len(db))
            minu = rng.choice([0, 5, 10, 20, 40, 60, 80])
            maxc = rng.choice([5, 10, 15, 20, 30, 50, 1000])
            exp = oracle(db, mins, minu, maxc)
            got = run_port(path, mins, minu, maxc, out)
            total_patterns += len(exp)
            m = set(exp) - set(got)
            x = set(got) - set(exp)
            w = [p for p in set(exp) & set(got)
                 if exp[p][0] != got[p][0] or abs(exp[p][1] - got[p][1]) > 1e-4 or abs(exp[p][2] - got[p][2]) > 1e-4]
            if m or x or w:
                bad += 1; missing += len(m); extra += len(x); wrong += len(w)
                if shown < 3:
                    shown += 1
                    print(f"MISMATCH case={case} mins={mins} minu={minu} maxc={maxc}")
                    print("  db:\n   " + to_file(db).replace("\n", "\n   "))
                    for p in sorted(m, key=len)[:3]:
                        print(f"  missing {fmt(p)} oracle={exp[p]}")
                    for p in sorted(x, key=len)[:3]:
                        print(f"  extra   {fmt(p)} port={got[p]}")
                    for p in w[:3]:
                        print(f"  values  {fmt(p)} oracle={exp[p]} port={got[p]}")
            if (case + 1) % 100 == 0:
                print(f"{case + 1} cases, mismatching {bad}", flush=True)
    row = {"algo": os.environ.get("ALGO", "flchuspm"), "long": LONG, "cases": CASES, "seed": SEED, "env": " ".join(f"{k}={v}" for k, v in EXTRA_ENV.items()),
           "oracle_patterns": total_patterns, "mismatching_cases": bad, "missing": missing,
           "extra": extra, "wrong_values": wrong}
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row)); wr.writeheader(); wr.writerow(row)
    print(row)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
