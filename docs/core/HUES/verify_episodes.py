"""Exact verifier for reported high utility episodes on large sequences (independent of hue_span.c).

For every episode in an SPMF HUE output file, the utility of Definition 12 of HUE-Span
(Fournier-Viger et al., ADMA 2019) is recomputed from the raw input:
  - occurrences start at a time point ts where SE_1 occurs; the earliest end te(ts) is found
    greedily (each SE_i at the first time point after the previous one where it occurs);
  - [ts, te(ts)] is a minimal occurrence iff no later start has an end <= te(ts); since
    te(.) is non-decreasing over starts, it is enough to compare with the next start; ends
    beyond ts + maxDur - 1 are treated as infinite, which is exact because such an
    occurrence (and every later one) ends after any te(ts) with te(ts) - ts + 1 <= maxDur;
  - the utility of a minimal occurrence is the maximum of sum_i u(SE_i, t_i) over matchings
    ts = t_1 < ... < t_k = te (reading I1 of Definition 12; on all oracle tests reading I2
    gave the same values), computed by dynamic programming.
The verifier only checks reported episodes: it detects wrong utilities and false HUEs, not
missing ones.

Usage (from the repository root):
  python3 docs/core/HUES/verify_episodes.py <input> <hue_output> <minUtilRatio> <maxDur> [csv_out]
"""
import csv
import sys


def read_sequence(path):
    """one time point per physical line, numbered from 1 (as the reference implementation)"""
    seq = [None]
    total = 0
    for line in open(path):
        line = line.strip()
        if not line or line[0] in "#%@":
            seq.append({})
            continue
        evs, tu, us = line.split(":")
        tp = {}
        for e, u in zip(evs.split(), us.split()):
            tp[int(e)] = tp.get(int(e), 0) + int(u)
        seq.append(tp)
        total += int(tu)
    return seq, total


def read_output(path):
    res = []
    for line in open(path):
        if line.startswith("@") or not line.strip():
            continue
        body, util = line.split("#UTIL:")
        sets, cur = [], []
        for tok in body.split():
            if tok == "-1":
                sets.append(tuple(sorted(cur)))
                cur = []
            else:
                cur.append(int(tok))
        res.append((tuple(sets), int(util)))
    return res


def u_set(tp, se):
    s = 0
    for e in se:
        if e not in tp:
            return None
        s += tp[e]
    return s


def exact_utility(seq, ep, max_dur, starts_of):
    n = len(seq) - 1
    k = len(ep)
    starts = starts_of(ep[0])
    INF = float("inf")
    ends = []                                          # earliest end, INF if beyond ts + maxDur - 1
    for ts in starts:
        t, i, lim = ts, 1, min(n, ts + max_dur - 1)
        while i < k and t < lim:
            t += 1
            if u_set(seq[t], ep[i]) is not None:
                i += 1
        ends.append(t if i == k else INF)
    total = 0
    count = 0
    for j, ts in enumerate(starts):
        te = ends[j]
        if te == INF:
            continue
        # te(.) is non-decreasing over starts, so only the next start can end no later
        if j + 1 < len(starts) and ends[j + 1] <= te:
            continue                                   # not minimal
        if k == 1:
            total += u_set(seq[ts], ep[0])
            count += 1
            continue
        neg = float("-inf")
        best = [neg] * (k + 1)
        best[1] = u_set(seq[ts], ep[0])
        for t in range(ts + 1, te + 1):
            for i in range(min(k, t - ts + 1), 1, -1):
                if best[i - 1] == neg:
                    continue
                if i == k and t != te:
                    continue
                u = u_set(seq[t], ep[i - 1])
                if u is not None and best[i - 1] + u > best[i]:
                    best[i] = best[i - 1] + u
        total += best[k]
        count += 1
    return total, count


def main():
    inp, outp, ratio, max_dur = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    csv_out = sys.argv[5] if len(sys.argv) > 5 else None
    seq, total = read_sequence(inp)
    min_abs = total * ratio
    index = {}
    for t in range(1, len(seq)):
        for e in seq[t]:
            index.setdefault(e, []).append(t)

    def starts_of(se):
        cand = min((index.get(e, []) for e in se), key=len)
        return [t for t in cand if u_set(seq[t], se) is not None]

    rows = []
    false_hue = wrong = 0
    for ep, rep in read_output(outp):
        exact, n_mo = exact_utility(seq, ep, max_dur, starts_of)
        is_false = exact < min_abs
        false_hue += is_false
        wrong += exact != rep
        rows.append({"episode": " -> ".join("(" + " ".join(map(str, s)) + ")" for s in ep), "length": len(ep),
                     "reported": rep, "exact": exact, "minimal_occurrences": n_mo,
                     "status": "false_hue" if is_false else ("wrong_utility" if exact != rep else "ok")})
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["episode"])
            w.writeheader()
            w.writerows(rows)
    print(f"input={inp} total={total} minUtilAbs={min_abs:.1f} maxDur={max_dur} reported={len(rows)} "
          f"wrong_utility={wrong} false_hue={false_hue}")


if __name__ == "__main__":
    main()
