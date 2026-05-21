"""
Python port of src/tokenizer/unigram_subword.c  train  command.
Implements Kudo 2018 Unigram LM tokenizer training with EM + shrinking.

Usage:
    python train_unigram.py -i <corpus...> --vocab-size N -o model.txt
                            [--seed-size N] [--max-piece-length N]
                            [--min-frequency N] [--shrinking-factor F]
                            [--em-iterations N] [--stats]
"""

import sys
import math
import argparse
from collections import defaultdict

LOG_ZERO = -1.0e100
UNI_SEP = "@@"


def read_words(paths):
    vocab = defaultdict(int)
    sources = [open(p, encoding="utf-8", errors="replace") for p in paths] if paths else [sys.stdin]
    for fh in sources:
        for line in fh:
            for word in line.split():
                vocab[word] += 1
        if paths:
            fh.close()
    return vocab


def codepoints(s):
    """Split string into Unicode codepoint strings (mirrors utf8_len logic)."""
    return list(s)


def build_seed(word_freq, max_len, seed_size, min_freq):
    """Enumerate all substrings up to max_len, return initial model + char set."""
    sub_counts = defaultdict(int)
    chars = set()
    for word, freq in word_freq.items():
        cps = codepoints(word)
        for i in range(len(cps)):
            chars.add(cps[i])
            for j in range(i + 1, min(i + max_len, len(cps)) + 1):
                sub = "".join(cps[i:j])
                sub_counts[sub] += freq

    # Add pieces with freq >= min_freq (chars always included)
    model = {}
    for piece, cnt in sorted(sub_counts.items(), key=lambda x: -x[1]):
        if piece in chars or cnt >= min_freq:
            model[piece] = float(cnt)
        if len(model) >= seed_size:
            break

    # Guarantee all chars are present
    for c in chars:
        if c not in model:
            model[c] = 1.0

    _normalize(model)
    return model, chars


def _normalize(model):
    z = sum(model.values())
    if z <= 0.0:
        z = 1.0
    for k in model:
        model[k] /= z


def _logsum2(a, b):
    if a <= LOG_ZERO / 2:
        return b
    if b <= LOG_ZERO / 2:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _build_lattice(model, word, alpha=1.0):
    """Returns (cps, lat) where lat[e] = list of (start, piece, log_prob)."""
    cps = codepoints(word)
    n = len(cps)
    max_piece_cp = max((len(codepoints(p)) for p in model), default=1)
    lat = [[] for _ in range(n + 1)]
    for st in range(n):
        for e in range(st + 1, min(st + max_piece_cp, n) + 1):
            sub = "".join(cps[st:e])
            p = model.get(sub, 0.0)
            if p > 0.0:
                lat[e].append((st, sub, math.log(p) * alpha))
    return cps, lat


def em_step(model, word_freq):
    """One forward-backward EM step. Returns (new_model, expected_total)."""
    new_counts = defaultdict(float)
    expected_total = 0.0

    for word, freq in word_freq.items():
        cps, lat = _build_lattice(model, word)
        n = len(cps)
        fwd = [LOG_ZERO] * (n + 1)
        bwd = [LOG_ZERO] * (n + 1)
        fwd[0] = 0.0
        bwd[n] = 0.0

        for e in range(1, n + 1):
            for (st, piece, lp) in lat[e]:
                fwd[e] = _logsum2(fwd[e], fwd[st] + lp)

        for st in range(n - 1, -1, -1):
            for e in range(st + 1, n + 1):
                for (s2, piece, lp) in lat[e]:
                    if s2 == st:
                        bwd[st] = _logsum2(bwd[st], lp + bwd[e])

        z = fwd[n]
        if z <= LOG_ZERO / 2:
            for c in cps:
                new_counts[c] += freq
                expected_total += freq
        else:
            for e in range(1, n + 1):
                for (st, piece, lp) in lat[e]:
                    post = math.exp(fwd[st] + lp + bwd[e] - z) * freq
                    new_counts[piece] += post
                    expected_total += post

    total = sum(new_counts.values())
    if total <= 0.0:
        total = 1.0
    return {k: v / total for k, v in new_counts.items()}, expected_total


def train_unigram(word_freq, vocab_size, seed_size, max_piece_len, min_freq, shrinking, em_iters):
    model, chars = build_seed(word_freq, max_piece_len, max(seed_size, vocab_size), min_freq)
    expected_total = 0.0

    while len(model) > vocab_size:
        for _ in range(em_iters):
            model, expected_total = em_step(model, word_freq)

        target = max(int(len(model) * shrinking), vocab_size)

        # Score non-char pieces: higher score = lower prob = candidate for pruning
        scored = [
            (piece, -math.log(max(prob, 1e-300)))
            for piece, prob in model.items()
            if piece not in chars
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))  # descending score

        keep = set(chars)
        budget = max(target - len(keep), 0)
        for piece, _ in scored[:budget]:
            keep.add(piece)

        if len(keep) >= len(model):
            break

        model = {k: v for k, v in model.items() if k in keep}
        _normalize(model)

    # Final EM pass
    for _ in range(max(em_iters, 1)):
        model, expected_total = em_step(model, word_freq)

    # Final trim to vocab_size
    if len(model) > vocab_size:
        sorted_pieces = sorted(model.items(), key=lambda x: (-x[1], x[0]))
        keep = set(chars)
        for piece, _ in sorted_pieces:
            if len(keep) >= vocab_size:
                break
            keep.add(piece)
        model = {k: v for k, v in model.items() if k in keep}
        _normalize(model)

    return model, expected_total


def write_model(model, path=None):
    fh = open(path, "w", encoding="utf-8") if path else sys.stdout
    fh.write("#version: dm-unigram-kudo-2018\n")
    for piece, prob in sorted(model.items(), key=lambda x: (-x[1], x[0])):
        fh.write(f"{piece}\t{prob:.17g}\n")
    if path:
        fh.close()


def main():
    ap = argparse.ArgumentParser(description="Unigram LM tokenizer trainer (mirrors unigram_subword.c train)")
    ap.add_argument("-i", "--input", nargs="+", metavar="FILE")
    ap.add_argument("-o", "--output", metavar="FILE", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--seed-size", type=int, default=8000)
    ap.add_argument("--max-piece-length", type=int, default=16)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--shrinking-factor", type=float, default=0.8)
    ap.add_argument("--em-iterations", type=int, default=2)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    word_freq = read_words(args.input)
    model, expected = train_unigram(
        word_freq,
        args.vocab_size,
        args.seed_size,
        args.max_piece_length,
        args.min_frequency,
        args.shrinking_factor,
        args.em_iterations,
    )
    write_model(model, args.output)
    if args.stats:
        print(
            f'{{"word_types":{len(word_freq)},"vocab_size":{len(model)},"expected_pieces":{expected:.12g}}}',
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
