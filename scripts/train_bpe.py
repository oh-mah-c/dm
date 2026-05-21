"""
Python port of src/tokenizer/bpe_subword.c  learn-bpe  command.
Produces the same output format: codes file + optional vocab file.

Usage:
    python train_bpe.py -i <corpus> -m <num_merges> [-o <codes_out>]
                        [--min-frequency N] [--vocab-out <path>] [--stats]
"""

import sys
import re
import argparse
from collections import defaultdict

EOW = "</w>"
SEPARATOR = "@@"

# ---------------------------------------------------------------------------
# Corpus reading
# ---------------------------------------------------------------------------

def read_words(paths):
    """Returns {word: freq} from one or more text files (or stdin)."""
    vocab = defaultdict(int)
    sources = [open(p, encoding="utf-8") for p in paths] if paths else [sys.stdin]
    for fh in sources:
        for line in fh:
            for word in line.split():
                vocab[word] += 1
        if paths:
            fh.close()
    return vocab

# ---------------------------------------------------------------------------
# Symbol representation  (word -> tuple of unicode chars + EOW)
# ---------------------------------------------------------------------------

def word_to_symbols(word):
    """Split a word into a list of UTF-8 codepoints with EOW appended."""
    return list(word) + [EOW]

def build_symbol_vocab(word_freq):
    """Returns list of (symbols_list, freq) for every word."""
    return [(word_to_symbols(w), f) for w, f in word_freq.items()]

# ---------------------------------------------------------------------------
# BPE training
# ---------------------------------------------------------------------------

def collect_pair_stats(sym_vocab):
    stats = defaultdict(int)
    for symbols, freq in sym_vocab:
        for i in range(len(symbols) - 1):
            stats[(symbols[i], symbols[i + 1])] += freq
    return stats

def best_pair(stats, min_freq):
    if not stats:
        return None
    best = max(stats, key=lambda p: (stats[p], p))
    if stats[best] < min_freq:
        return None
    return best

def apply_merge(sym_vocab, left, right):
    joined = left + right
    new_vocab = []
    for symbols, freq in sym_vocab:
        new_syms = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                new_syms.append(joined)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        new_vocab.append((new_syms, freq))
    return new_vocab

def learn_bpe(word_freq, num_merges, min_freq=2):
    """
    Returns:
        merges   - list of (left, right) in order applied
        sym_counts - {token: total_count} after all merges
    """
    sym_vocab = build_symbol_vocab(word_freq)

    # initial character counts
    sym_counts = defaultdict(int)
    for symbols, freq in sym_vocab:
        for s in symbols:
            sym_counts[s] += freq

    merges = []

    for _ in range(num_merges):
        stats = collect_pair_stats(sym_vocab)
        pair = best_pair(stats, min_freq)
        if pair is None:
            break
        left, right = pair
        merges.append((left, right))
        joined = left + right
        sym_counts[joined] += stats[pair]
        sym_vocab = apply_merge(sym_vocab, left, right)

    return merges, sym_counts

# ---------------------------------------------------------------------------
# Output writers  (match bpe_subword.c format exactly)
# ---------------------------------------------------------------------------

def write_merges(merges, path=None):
    fh = open(path, "w", encoding="utf-8") if path else sys.stdout
    fh.write("#version: dm-bpe-sennrich-2016\n")
    for left, right in merges:
        fh.write(f"{left} {right}\n")
    if path:
        fh.close()

def write_vocab(sym_counts, path=None):
    fh = open(path, "w", encoding="utf-8") if path else sys.stdout
    for token, count in sorted(sym_counts.items(), key=lambda x: (-x[1], x[0])):
        fh.write(f"{token}\t{count}\n")
    if path:
        fh.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="BPE trainer (mirrors bpe_subword.c learn-bpe)")
    ap.add_argument("-i", "--input", nargs="+", metavar="FILE", help="corpus files (default: stdin)")
    ap.add_argument("-m", "--merges", type=int, required=True, metavar="N", help="number of merge operations")
    ap.add_argument("-o", "--output", metavar="FILE", help="codes output file (default: stdout)")
    ap.add_argument("--min-frequency", type=int, default=2, metavar="N")
    ap.add_argument("--vocab-out", metavar="FILE", help="optional vocab output file")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    word_freq = read_words(args.input)
    merges, sym_counts = learn_bpe(word_freq, args.merges, args.min_frequency)

    write_merges(merges, args.output)
    if args.vocab_out:
        write_vocab(sym_counts, args.vocab_out)
    if args.stats:
        total_tokens = sum(word_freq.values())
        print(
            f'{{"merges": {len(merges)}, "symbol_types": {len(sym_counts)}, '
            f'"word_tokens": {total_tokens}, "word_types": {len(word_freq)}}}',
            file=sys.stderr,
        )

if __name__ == "__main__":
    main()
