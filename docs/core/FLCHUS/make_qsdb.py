"""Build the quantitative sequence databases (QSDBs) used to evaluate FLCHUSM algorithms.

Truong et al. (2022), Section 6: "utilities and costs of items were generated respectively in
the [10, 100] and [1, 70] intervals using a log-normal distribution". The parameters of the
log-normal distribution are not given; this script uses, independently for every q-item
(occurrence), v = round(exp(Y)) with Y ~ Normal(mu, sigma) truncated to [ln lo, ln hi],
mu = (ln lo + ln hi) / 2 and sigma = (ln hi - ln lo) / 6 (the interval is +-3 sigma).
Generation is seeded, so the files are reproducible.

Inputs (SPMF public datasets, https://www.philippe-fournier-viger.com/spmf/publicdatasets/):
  BMS1_spmf.txt, SIGN.txt, FIFA.txt, LEVIATHAN.txt, BIBLE.txt (SPMF sequence format,
  "-1" closes an itemset, "-2" closes a sequence) and snake.dat (protein sequences: the 163
  lines made only of upper-case letters with length >= 57 are the sequences of Table 3 of the
  paper; each letter is one itemset with one item).
Output: one q-sequence per line, q-items "item:cost:utility", "-1", "-2".

  python3 -I docs/core/FLCHUS/make_qsdb.py <input_dir> <output_dir>
"""
import math
import os
import random
import sys

SEED = 2022
DATASETS = ["BMS1_spmf.txt", "SIGN.txt", "FIFA.txt", "LEVIATHAN.txt", "BIBLE.txt", "snake.dat"]


def lognormal_in(rng, lo, hi):
    a, b = math.log(lo), math.log(hi)
    mu, sigma = (a + b) / 2, (b - a) / 6
    while True:
        y = rng.gauss(mu, sigma)
        if a <= y <= b:
            return max(lo, min(hi, round(math.exp(y))))


def read_spmf(path):
    seqs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        t = line.split()
        if not t or t[0][0] in "#@%":
            continue
        seq, cur = [], []
        for x in t:
            if x == "-1":
                if cur:
                    seq.append(sorted(set(int(v) for v in cur)))
                cur = []
            elif x == "-2":
                break
            else:
                cur.append(x)
        if cur:
            seq.append(sorted(set(int(v) for v in cur)))
        seqs.append(seq)
    return seqs


def read_snake(path):
    letters = {}
    seqs = []
    for line in open(path, encoding="ascii", errors="replace"):
        s = line.strip()
        if len(s) >= 57 and s.isalpha() and s.isupper():
            seqs.append([[letters.setdefault(ch, len(letters) + 1)] for ch in s])
    return seqs


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for name in DATASETS:
        rng = random.Random(f"{SEED}:{name}")
        seqs = read_snake(os.path.join(src, name)) if name.endswith(".dat") else read_spmf(os.path.join(src, name))
        base = name.split(".")[0].replace("_spmf", "")
        out = os.path.join(dst, f"{base}_qsdb.txt")
        items = set()
        with open(out, "w") as f:
            for seq in seqs:
                toks = []
                for its in seq:
                    for i in its:
                        items.add(i)
                        toks.append(f"{i}:{lognormal_in(rng, 1, 70)}:{lognormal_in(rng, 10, 100)}")
                    toks.append("-1")
                toks.append("-2")
                f.write(" ".join(toks) + "\n")
        lens = [sum(len(s) for s in q) for q in seqs]
        print(f"{base}: sequences={len(seqs)} items={len(items)} avg_len={sum(lens) / len(lens):.1f} "
              f"min={min(lens)} max={max(lens)} -> {out}")


if __name__ == "__main__":
    main()
