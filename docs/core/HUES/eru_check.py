"""Falsification test of the ERU upper bound of HUE-Span (Definition 18 / Lemma 1, ADMA 2019) with
the utility of Definition 12 (maximum over matchings, reading I1).

For random small sequences, every episode beta with an occurrence of length <= maxDur and every
search-tree ancestor alpha of beta are enumerated; the test checks u(beta) <= ERU(alpha), where

  ERU(alpha) = sum over minimal occurrences [ts, te] of alpha with te - ts + 1 <= maxDur of
               u(alpha, [ts, te]) + sum_{x in tSE_te, x > SE_k} u(x, te)
               + sum_{j = te+1}^{ts+maxDur-1} u(tSE_j)

with events totally ordered by identifier. alpha is an ancestor of beta = <B_1..B_m> if
alpha = <B_1, ..., B_{j-1}, P> where P is a non-empty initial segment (in identifier order)
of B_j, and alpha != beta. Prints the first counterexamples found.

  python3 docs/core/HUES/eru_check.py
"""
import importlib.util
import os
import random

spec = importlib.util.spec_from_file_location("o", os.path.join(os.path.dirname(__file__), "oracle_huespan.py"))
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)

SEQUENCES = int(os.environ.get("SEQUENCES", "400"))
MAXDURS = [int(x) for x in os.environ.get("MAXDURS", "1,2,3,4").split(",")]


def minimal_occurrences(seq, ep, max_dur):
    n = len(seq)
    occ = {}
    for ts in range(n):
        for te in range(ts, min(n, ts + max_dur)):
            m = o.matchings(seq, ep, ts, te)
            if m:
                occ[(ts, te)] = m
                break                                    # earliest end only
    # an occurrence longer than maxDur can never be inside a short one, so the earliest-end
    # occurrences found within maxDur are minimal iff no other one lies inside them
    res = []
    for (ts, te), m in occ.items():
        if any((a, b) != (ts, te) and ts <= a and b <= te for (a, b) in occ):
            continue
        res.append((ts, te, m))
    return res


def eru(seq, ep, max_dur):
    n = len(seq)
    last = max(ep[-1])
    tot = 0
    for ts, te, m in minimal_occurrences(seq, ep, max_dur):
        u = max(sum(o.u_set(seq, ep[i], t[i]) for i in range(len(ep))) for t in m)
        r = sum(v for x, v in seq[te].items() if x > last)
        w = sum(sum(seq[j].values()) for j in range(te + 1, min(n, ts + max_dur)))
        tot += u + r + w
    return tot


def ancestors(beta):
    for j in range(len(beta)):
        s = sorted(beta[j])
        for k in range(1, len(s) + 1):
            alpha = tuple(beta[:j]) + (frozenset(s[:k]),)
            if alpha != tuple(beta):
                yield alpha


def main():
    rng = random.Random(31)
    checked = viol = 0
    shown = 0
    for _ in range(SEQUENCES):
        seq = o.make_sequence(rng)
        for md in MAXDURS:
            cache = {}
            for beta in o.candidate_episodes(seq, md):
                ub = o.utility(seq, beta, md, "I1")
                for alpha in ancestors(beta):
                    if alpha not in cache:
                        cache[alpha] = eru(seq, alpha, md)
                    checked += 1
                    if ub > cache[alpha]:
                        viol += 1
                        if shown < 3:
                            shown += 1
                            print(f"VIOLATION maxDur={md} alpha={o.fmt(alpha)} ERU={cache[alpha]} "
                                  f"beta={o.fmt(beta)} u={ub}")
                            for i, tp in enumerate(seq, 1):
                                print(f"   t{i}: " + " ".join(f"{chr(64 + e)}({u})" for e, u in sorted(tp.items())))
    print(f"pairs checked {checked}, violations {viol}")


if __name__ == "__main__":
    main()
