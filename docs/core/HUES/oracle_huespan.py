"""Brute-force oracle for the redefined high utility episode mining problem of HUE-Span
(Fournier-Viger et al., ADMA 2019, Definitions 2-4 and 12-14), compared with the reference
implementation (dm.exe hue_span, a parity-checked port of the SPMF Java code).

Problem, as printed in the paper:
  - an episode is <SE_1, ..., SE_k> with non-empty event sets; SE_i occurs at t if SE_i is a
    subset of the events at t;
  - [ts, te] is an occurrence if there are time points ts = t_1 < ... < t_k = te with SE_i
    occurring at t_i; it is minimal if no other occurrence is a sub-interval of it;
  - only minimal occurrences with te - ts + 1 <= maxDur count;
  - utility of a minimal occurrence (Definition 12), two readings:
      I1: maximum of sum_i u(SE_i, t_i) over all order-respecting matchings with t_1 = ts, t_k = te;
      I2: u(SE_1, ts) + sum over middle positions of the maximum of u(SE_i, t) over every t at
          which SE_i is matched in some order-respecting matching, + u(SE_k, te);
  - u(alpha) = sum over minimal occurrences; HUE iff u(alpha) >= minUtil * (total utility).

For random small sequences (no comment lines), all episodes that have an occurrence of length
<= maxDur are enumerated, and the set of HUEs (with utilities) under I1 and under I2 is compared
with the output of hue_span (default SPMF configuration, single events output).

Run from the repository root:
  python3 docs/core/HUES/oracle_huespan.py -> docs/core/HUES/results/oracle_huespan.csv
"""
import csv
import itertools
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

BIN = "./build/bin/dm.exe"
OUT = Path(os.environ.get("ORACLE_OUT", "docs/core/HUES/results/oracle_huespan.csv"))
SEQUENCES = int(os.environ.get("SEQUENCES", "300"))
SEED = 12
# events of each line in increasing identifier order (SPMF files are written this way); set
# SORTED_LINES=0 to keep a random order inside lines
SORTED_LINES = os.environ.get("SORTED_LINES", "1") == "1"


def make_sequence(rng):
    n_ev = rng.randint(2, 4)
    L = rng.randint(2, 9)
    seq = []
    for _ in range(L):
        k = rng.randint(1, n_ev)
        evs = rng.sample(range(1, n_ev + 1), k)
        seq.append({e: rng.randint(1, 9) for e in evs})
    return seq


def to_spmf(seq):
    lines = []
    for tp in seq:
        evs = sorted(tp) if SORTED_LINES else list(tp)
        lines.append(f"{' '.join(map(str, evs))}:{sum(tp.values())}:{' '.join(str(tp[e]) for e in evs)}")
    return "\n".join(lines) + "\n"


def nonempty_subsets(s):
    s = sorted(s)
    for k in range(1, len(s) + 1):
        for c in itertools.combinations(s, k):
            yield frozenset(c)


def candidate_episodes(seq, max_dur):
    """every episode that has at least one occurrence of length <= max_dur"""
    n = len(seq)
    cands = set()
    for ts in range(n):
        last = min(n - 1, ts + max_dur - 1)
        rest = list(range(ts + 1, last + 1))
        for r in range(len(rest) + 1):
            for later in itertools.combinations(rest, r):
                tps = (ts,) + later
                for sets in itertools.product(*(list(nonempty_subsets(seq[t])) for t in tps)):
                    cands.add(tuple(sets))
    return cands


def u_set(seq, SE, t):
    return sum(seq[t][e] for e in SE)


def matchings(seq, ep, ts, te):
    """all order-respecting matchings (tuples of time points) with t_1 = ts and t_k = te"""
    k = len(ep)
    if not ep[0] <= seq[ts].keys() or not ep[-1] <= seq[te].keys():
        return []
    if k == 1:
        return [(ts,)] if ts == te else []
    if ts >= te:
        return []
    res = []

    def rec(i, prev, acc):
        if i == k - 1:
            res.append(acc + (te,))
            return
        for t in range(prev + 1, te):
            if ep[i] <= seq[t].keys():
                rec(i + 1, t, acc + (t,))

    rec(1, ts, (ts,))
    return res


def utility(seq, ep, max_dur, reading):
    n = len(seq)
    occ = {}
    for ts in range(n):
        for te in range(ts, n):
            m = matchings(seq, ep, ts, te)
            if m:
                occ[(ts, te)] = m
    total = 0
    for (ts, te), m in occ.items():
        if any((a, b) != (ts, te) and ts <= a and b <= te for (a, b) in occ):
            continue                               # not minimal
        if te - ts + 1 > max_dur:
            continue
        if reading == "I1":
            total += max(sum(u_set(seq, ep[i], t[i]) for i in range(len(ep))) for t in m)
        else:
            k = len(ep)
            val = u_set(seq, ep[0], ts) + (u_set(seq, ep[-1], te) if k > 1 else 0)
            for i in range(1, k - 1):
                val += max(u_set(seq, ep[i], t[i]) for t in m)
            total += val
    return total


def oracle(seq, ratio, max_dur, reading):
    min_abs = sum(sum(tp.values()) for tp in seq) * ratio
    res = {}
    for ep in candidate_episodes(seq, max_dur):
        u = utility(seq, ep, max_dur, reading)
        if u >= min_abs:
            res[ep] = u
    return res


def run_port(path, ratio, max_dur, out):
    if os.path.exists(out):
        os.remove(out)
    subprocess.run([BIN, "hue_span", path, "0", str(ratio), str(max_dur), "true"], capture_output=True,
                   text=True, timeout=120, env=dict(os.environ, DM_HUE_OUT=out), check=True)
    res = {}
    for line in open(out):
        if line.startswith("@") or not line.strip():
            continue
        body, util = line.split("#UTIL:")
        sets, cur = [], []
        for tok in body.split():
            if tok == "-1":
                sets.append(frozenset(cur))
                cur = []
            else:
                cur.append(int(tok))
        res[tuple(sets)] = int(util)
    return res


def fmt(ep):
    return " ".join("(" + "".join(chr(64 + e) for e in sorted(s)) + ")" for s in ep)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    cases = 0
    stats = {"I1": [0, 0, 0], "I2": [0, 0, 0]}   # mismatching cases, missing episodes, wrong-utility/extra
    examples = []
    with tempfile.TemporaryDirectory() as tmp:
        path, out = os.path.join(tmp, "seq.txt"), os.path.join(tmp, "out.txt")
        for s in range(SEQUENCES):
            seq = make_sequence(rng)
            open(path, "w").write(to_spmf(seq))
            for _ in range(2):
                ratio, max_dur = rng.choice([0.05, 0.1, 0.2, 0.3, 0.5]), rng.choice([1, 2, 3, 4])
                got = run_port(path, ratio, max_dur, out)
                cases += 1
                for reading in ("I1", "I2"):
                    exp = oracle(seq, ratio, max_dur, reading)
                    if exp != got:
                        st = stats[reading]
                        st[0] += 1
                        st[1] += len(set(exp) - set(got))
                        st[2] += len([e for e in got if e not in exp or exp[e] != got[e]])
                        if reading == "I1" and len(examples) < 5:
                            miss = sorted(set(exp) - set(got), key=len)[:3]
                            extra = [e for e in got if e not in exp][:3]
                            wrong = [e for e in got if e in exp and exp[e] != got[e]][:3]
                            examples.append((seq, ratio, max_dur, miss, extra, wrong, exp, got))
            if (s + 1) % 50 == 0:
                print(f"{s + 1} sequences, {cases} cases, I1 mismatching cases {stats['I1'][0]}, "
                      f"I2 mismatching cases {stats['I2'][0]}", flush=True)
    for seq, ratio, max_dur, miss, extra, wrong, exp, got in examples:
        print(f"--- example: minUtil={ratio} maxDur={max_dur} total={sum(sum(tp.values()) for tp in seq)}")
        for i, tp in enumerate(seq, 1):
            print(f"   t{i}: " + " ".join(f"{chr(64 + e)}({u})" for e, u in sorted(tp.items())))
        for e in miss:
            print(f"   missing {fmt(e)} oracle u={exp[e]}")
        for e in extra:
            print(f"   extra   {fmt(e)} reported u={got[e]}")
        for e in wrong:
            print(f"   utility {fmt(e)} oracle={exp[e]} reported={got[e]}")
    rows = [{"reading": r, "sequences": SEQUENCES, "seed": SEED, "cases": cases, "mismatching_cases": v[0],
             "missing_episodes": v[1], "extra_or_wrong_utility": v[2]} for r, v in stats.items()]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
