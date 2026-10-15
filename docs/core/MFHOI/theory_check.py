"""Brute-force check of every statement of the theory of the MHOI paper.

On random databases (mixed lengths, equal lengths, and with items that occur
once), with exact rational arithmetic, this script checks:
  L1   occ(X) <= supp(X), and every HOI has support >= ceil(xi)
  L2   occ(cl X) = |cl X| / |X| * occ(X) >= occ(X)
  T3   every MHOI is closed, and MHOI = Max(HOI intersected with closed frequent itemsets)
  T4a  every maximal frequent itemset (support ceil(xi)) that is an HOI is an MHOI
  T4c  with equal lengths L: X is an HOI iff supp(X) >= xi * L / |X|
  P5   with equal lengths in the cover: UBO(X) = supp(X)
  Deng occ(Y) <= UBO(X) for every superset Y of X (Theorem 4 of Deng 2020)
  T6   for every node C of the prefix-preserving closure-extension tree and its
       tail R: occ(Y) <= B(C) for every HOI Y = C + Z, Z a non-empty subset of R
  T7   B(C) <= UBO(C)
  C8   occ(Y) <= B_e(C) for every HOI Y = C + {e} + Z', Z' a subset of R after e
  TREE every descendant of C in the tree is contained in C + R(C)
and the two counterexamples of Theorem 4(b). Exit status 1 on any violation.

  python3 docs/core/MFHOI/theory_check.py [databases per suite]
"""
import itertools
import math
import random
import sys
from fractions import Fraction as F

THRESHOLDS = (F(1, 2), F(1), F(3, 2), F(2), F(5, 2), F(3), F(4))
fails = []
checks = {}


def ok(name, cond, info):
    checks[name] = checks.get(name, 0) + 1
    if not cond:
        fails.append((name, info))


def analyse(D, xi):
    n = len(D)
    items = sorted(set().union(*D))
    cover = lambda X: [q for q in range(n) if X <= D[q]]
    occ = lambda X: len(X) * sum(F(1, len(D[q])) for q in cover(X))

    def cl(X):
        c = cover(X)
        return frozenset(set.intersection(*[set(D[q]) for q in c]))

    def ubo(X):
        c = cover(X)
        return max(l * sum(F(1, len(D[q])) for q in c if len(D[q]) >= l) for l in {len(D[q]) for q in c})

    allX = [frozenset(c) for k in range(1, len(items) + 1) for c in itertools.combinations(items, k) if cover(frozenset(c))]
    sigma = math.ceil(xi)
    H = [X for X in allX if occ(X) >= xi]
    Hs = set(H)
    M = {X for X in H if not any(X < Y for Y in Hs)}
    FI = [X for X in allX if len(cover(X)) >= sigma]
    MFI = {X for X in FI if not any(X < Y for Y in FI)}
    closed_freq = {X for X in FI if cl(X) == X}
    ctx = str([sorted(t) for t in D]) + f" xi={xi}"

    for X in allX:
        ok("L1", occ(X) <= len(cover(X)), (ctx, sorted(X)))
        c = cl(X)
        ok("L2", occ(c) == F(len(c), len(X)) * occ(X) and occ(c) >= occ(X), (ctx, sorted(X)))
        for Y in allX:
            if X <= Y:
                ok("Deng-Thm4", occ(Y) <= ubo(X), (ctx, sorted(X), sorted(Y)))
        lens = {len(D[q]) for q in cover(X)}
        if len(lens) == 1:
            ok("P5", ubo(X) == len(cover(X)), (ctx, sorted(X)))
    for X in H:
        ok("L1-supp", len(cover(X)) >= sigma, (ctx, sorted(X)))
    for X in M:
        ok("T3-closed", cl(X) == X, (ctx, sorted(X)))
    HC = [X for X in H if X in closed_freq]
    ok("T3-max", M == {X for X in HC if not any(X < Y for Y in HC)}, ctx)
    for X in MFI:
        if occ(X) >= xi:
            ok("T4a", X in M, (ctx, sorted(X)))
    if len({len(t) for t in D}) == 1:
        L = len(D[0])
        for X in allX:
            ok("T4c", (occ(X) >= xi) == (len(cover(X)) >= xi * L / len(X)), (ctx, sorted(X)))

    # prefix-preserving closure-extension tree with items ordered by support
    sup = {i: len(cover(frozenset([i]))) for i in items}
    order = sorted([i for i in items if sup[i] >= sigma], key=lambda i: (sup[i], i))
    rank = {i: k for k, i in enumerate(order)}

    def bound(units, base, kmin):
        cmax = max((c for _, c in units), default=0)
        best = F(0)
        for k in range(kmin, cmax + 1):
            Q = [L for L, c in units if c >= k]
            if len(Q) >= sigma:
                best = max(best, (base + k) * sum(F(1, L) for L in Q))
        return best

    def node(C, core):
        cv = cover(C) if C else list(range(n))
        R = [i for i in order if (core is None or rank[i] > rank[core]) and i not in C
             and sum(1 for q in cv if i in D[q]) >= sigma]
        units = [(len(D[q]), len(set(R) & D[q])) for q in cv]
        B = bound(units, len(C), 1) if R else F(0)
        if C:
            ok("T7", B <= ubo(C), (ctx, sorted(C)))
        for k in range(1, len(R) + 1):
            for Z in itertools.combinations(R, k):
                Y = C | frozenset(Z)
                if cover(Y) and occ(Y) >= xi:
                    ok("T6", occ(Y) <= B, (ctx, sorted(C), sorted(Y)))
        for idx, e in enumerate(R):
            after = R[idx + 1:]
            ce = [q for q in cv if e in D[q]]
            Be = bound([(len(D[q]), len(set(after) & D[q])) for q in ce], len(C) + 1, 0)
            for k in range(0, len(after) + 1):
                for Z in itertools.combinations(after, k):
                    Y = C | {e} | frozenset(Z)
                    if cover(Y) and occ(Y) >= xi:
                        ok("C8", occ(Y) <= Be, (ctx, sorted(C), e, sorted(Y)))
        desc = []
        for e in R:
            Y = cl(C | {e})
            if any(rank.get(i, -1) < rank[e] and i not in C for i in Y):
                continue
            desc.append(Y)
            for Z in node(Y, e):
                desc.append(Z)
        for Y in desc:
            ok("TREE", Y <= C | frozenset(R), (ctx, sorted(C), sorted(Y)))
        return desc

    root = frozenset(set.intersection(*[set(t) for t in D]))
    if n >= sigma:
        visited = [root] + node(root, None)
        ok("TREE-closed", {X for X in visited if X} >= {X for X in closed_freq}, ctx)


def counterexamples():
    x = [f"x{i}" for i in range(7)]
    y = [f"y{i}" for i in range(7)]
    for D, xi, mhoi, mfi in (
        ([{"a", "b"}, {"a", "b"}, {"a", "b", "c", *x}, {"a", "b", "c", *y}], F(2), [{"a", "b"}], [{"a", "b", "c"}]),
        ([set("acd"), set("abd"), set("bcde"), set("ad"), set("cde"), set("abcde")], F(2),
         [set("ad"), set("cde")], [set("abd"), set("acd"), set("bcde")]),
        ([set("bde"), set("bcdef"), set("acdf"), set("af"), set("acdef"), set("abcdef"), set("abcde"), set("cef")], F(2),
         [set("acde"), set("acdf"), set("bcde"), set("cdef")], [set("abcde"), set("acdef"), set("bcdef")]),
        ([{"a", "b", "x1"}, {"a", "b", "x2"}, {"a", "c", "x3"}, {"a", "c", "x4"}, {"a", "d", "x5"}, {"a", "d", "x6"}],
         F(2), [{"a"}], [{"a", "b"}, {"a", "c"}, {"a", "d"}]),
    ):
        n = len(D)
        items = sorted(set().union(*D))
        cover = lambda X: [q for q in range(n) if X <= D[q]]
        occ = lambda X: len(X) * sum(F(1, len(D[q])) for q in cover(X))
        allX = [frozenset(c) for k in range(1, len(items) + 1) for c in itertools.combinations(items, k) if cover(set(c))]
        H = [X for X in allX if occ(X) >= xi]
        M = {X for X in H if not any(X < Y for Y in H)}
        FI = [X for X in allX if len(cover(X)) >= math.ceil(xi)]
        MFI = {X for X in FI if not any(X < Y for Y in FI)}
        ok("T4b-example", M == {frozenset(s) for s in mhoi} and MFI == {frozenset(s) for s in mfi} and not (M & MFI),
           (D, sorted(map(sorted, M)), sorted(map(sorted, MFI))))


def main():
    per_suite = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    counterexamples()
    rng = random.Random(7)
    dbs = 0
    for uniform, noise in ((False, 0), (True, 0), (False, 4)):
        for _ in range(per_suite):
            m = rng.randint(3, 7)
            n = rng.randint(3, 10)
            L = rng.randint(2, m)
            D = []
            for _ in range(n):
                k = L if uniform else rng.randint(1, m)
                t = set(rng.sample(range(1, m + 1), k)) | {100 + rng.randint(0, 999) for _ in range(rng.randint(0, noise))}
                D.append(t)
            dbs += 1
            for xi in THRESHOLDS:
                analyse(D, xi)
    print(f"databases={dbs} thresholds={len(THRESHOLDS)}")
    for k in sorted(checks):
        print(f"  {k:12s} checked {checks[k]:>9,d}")
    print(f"violations={len(fails)}")
    for f in fails[:10]:
        print("  ", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
