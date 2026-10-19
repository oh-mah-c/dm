"""Independent check of the number of HOIs on one dataset and threshold.

Enumerates every itemset with support >= ceil(alpha * n) (depth-first, TID sets
as Python integers used as bitsets) and evaluates occ(X) >= alpha * n exactly
with integer arithmetic. It shares no code with the C implementations.

  python3 docs/core/AURA-HOI/brute_check.py datasets/itemsets/accidents.txt 0.2
"""
import math
import sys
from fractions import Fraction


def main(path, alpha_s):
    trans = [sorted(set(map(int, line.split()))) for line in open(path) if line.split()]
    n = len(trans)
    alpha = Fraction(alpha_s)
    xi = alpha * n
    sigma = math.ceil(xi)
    lengths = sorted({len(t) for t in trans})
    M = math.lcm(*lengths)
    group = {L: 0 for L in lengths}            # bitset of the transactions of each length
    tids = {}
    for q, t in enumerate(trans):
        group[len(t)] |= 1 << q
        for i in t:
            tids[i] = tids.get(i, 0) | (1 << q)
    groups = [(M // L, g) for L, g in group.items()]
    target = xi * M                             # occ(X) >= xi  <=>  |X| * sum cnt_L * M/L >= xi * M

    items = sorted((i for i in tids if tids[i].bit_count() >= sigma), key=lambda i: (tids[i].bit_count(), i))
    frequent = hois = 0
    best = Fraction(0)
    found = []

    def visit(prefix, tid, k):
        nonlocal frequent, hois, best
        frequent += 1
        val = k * sum(w * (tid & g).bit_count() for w, g in groups)
        occ = Fraction(val, M)
        if occ > best:
            best = occ
        if val >= target:
            hois += 1
            found.append((list(prefix), occ))

    def dfs(prefix, tid, cands):
        for j, (i, ti) in enumerate(cands):
            t2 = tid & ti
            if t2.bit_count() < sigma:
                continue
            prefix.append(i)
            visit(prefix, t2, len(prefix))
            dfs(prefix, t2, [(i2, t3) for i2, t3 in cands[j + 1:]])
            prefix.pop()

    full = (1 << n) - 1
    dfs([], full, [(i, tids[i]) for i in items])
    print(f"n={n} alpha={alpha_s} xi={float(xi)} sigma={sigma}")
    print(f"frequent itemsets={frequent} HOIs={hois} max occ/n={float(best / n):.6f}")
    for X, occ in found[:20]:
        print("  HOI", sorted(X), f"occ/n={float(occ / n):.6f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
