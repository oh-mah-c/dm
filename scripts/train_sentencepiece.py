"""
Python port of src/tokenizer/sentencepiece_lite.c  train  command.
Supports --model-type bpe|unigram.  Produces a JSON model file.

Usage:
    python train_sentencepiece.py --input <files...> --model-type bpe|unigram
                                  --vocab-size N [-o model.json]
                                  [--normalization nfkc|nfc|lower|none]
                                  [--no-add-dummy-prefix]
                                  [--min-frequency N] [--seed-size N]
                                  [--max-piece-length N] [--write-vocab]
                                  [--stats]
"""

import sys
import json
import unicodedata
import argparse
from collections import defaultdict

SP_SPACE = "▁"   # ▁
SP_UNK   = "<unk>"
SP_BOS   = "<s>"
SP_EOS   = "</s>"
SP_PAD   = "<pad>"

SPECIAL_TOKENS = [SP_UNK, SP_BOS, SP_EOS, SP_PAD]


# ---------------------------------------------------------------------------
# Normalisation + whitespace escaping (mirrors normalize_escape in C)
# ---------------------------------------------------------------------------

def normalize_text(text, norm):
    if norm == "lower":
        return text.lower()
    if norm in ("nfkc", "nfc"):
        return unicodedata.normalize(norm.upper(), text)
    return text  # "none" or unknown


def escape_ws(text, add_dummy):
    result = (SP_SPACE if add_dummy else "") + text.replace(" ", SP_SPACE)
    return result


def normalize_escape(text, norm, add_dummy):
    return escape_ws(normalize_text(text, norm), add_dummy)


# ---------------------------------------------------------------------------
# Corpus reading (reads lines, not words — mirrors read_lines_normalized)
# ---------------------------------------------------------------------------

def read_lines(paths, norm, add_dummy):
    """Returns {normalized_line: freq}."""
    counts = defaultdict(int)
    sources = [open(p, encoding="utf-8", errors="replace") for p in paths] if paths else [sys.stdin]
    for fh in sources:
        for line in fh:
            line = line.rstrip("\n\r")
            ne = normalize_escape(line, norm, add_dummy)
            if ne:
                counts[ne] += 1
        if paths:
            fh.close()
    return counts


# ---------------------------------------------------------------------------
# BPE training (mirrors train_bpe)
# ---------------------------------------------------------------------------

def _collect_pairs_bpe(sym_corpus):
    stats = defaultdict(int)
    for syms, freq in sym_corpus:
        for i in range(len(syms) - 1):
            stats[(syms[i], syms[i + 1])] += freq
    return stats


def _best_pair_bpe(stats, min_freq):
    if not stats:
        return None
    best = max(stats, key=lambda p: (stats[p], p))
    return best if stats[best] >= min_freq else None


def _apply_merge_bpe(sym_corpus, left, right):
    joined = left + right
    new_corpus = []
    for syms, freq in sym_corpus:
        new_syms = []
        i = 0
        while i < len(syms):
            if i + 1 < len(syms) and syms[i] == left and syms[i + 1] == right:
                new_syms.append(joined)
                i += 2
            else:
                new_syms.append(syms[i])
                i += 1
        new_corpus.append((new_syms, freq))
    return new_corpus


def train_bpe(line_freq, reserved, vocab_size, min_freq):
    """Returns (vocab_list, merges_list)."""
    sym_corpus = []
    chars = set()
    for line, freq in line_freq.items():
        syms = list(line)      # split into codepoints
        for c in syms:
            chars.add(c)
        sym_corpus.append((syms, freq))

    chars = sorted(chars)
    target = max(vocab_size - len(reserved) - len(chars), 0)
    merges = []

    for _ in range(target):
        stats = _collect_pairs_bpe(sym_corpus)
        pair = _best_pair_bpe(stats, min_freq)
        if pair is None:
            break
        left, right = pair
        merges.append((left, right))
        sym_corpus = _apply_merge_bpe(sym_corpus, left, right)

    # Build vocab: reserved + chars + merged pieces
    vocab = list(reserved)
    for c in chars:
        if c not in vocab:
            vocab.append(c)
    for left, right in merges:
        piece = left + right
        if piece not in vocab:
            vocab.append(piece)
    vocab = vocab[:vocab_size]
    return vocab, merges


# ---------------------------------------------------------------------------
# Unigram (lite) training (mirrors train_unigram_lite — frequency-based, no EM)
# ---------------------------------------------------------------------------

def train_unigram_lite(line_freq, reserved, vocab_size, seed_size, max_len, min_freq):
    """Returns (vocab_list, pieces_with_probs)."""
    sub_counts = defaultdict(int)
    for line, freq in line_freq.items():
        cps = list(line)
        for i in range(len(cps)):
            for j in range(i + 1, min(i + max_len, len(cps)) + 1):
                sub = "".join(cps[i:j])
                sub_counts[sub] += freq

    target = max(vocab_size - len(reserved), 1)
    seed = max(seed_size, target)

    candidates = sorted(sub_counts.items(), key=lambda x: -x[1])[:seed]
    # Trim to target by frequency
    pieces_ranked = sorted(candidates, key=lambda x: -x[1])[:target]

    z = sum(cnt for _, cnt in pieces_ranked) or 1.0
    pieces = [(p, cnt / z) for p, cnt in pieces_ranked]

    # Build vocab
    vocab = list(reserved)
    for piece, _ in pieces:
        if piece not in vocab:
            vocab.append(piece)
        if len(vocab) >= vocab_size:
            break

    return vocab, pieces


# ---------------------------------------------------------------------------
# JSON model writer (mirrors write_model in sentencepiece_lite.c exactly)
# ---------------------------------------------------------------------------

def write_model(path, model_type, vocab, norm, add_dummy,
                unk, bos, eos, pad, merges, uni_pieces):
    data = {
        "version": "dm-sentencepiece-lite-D18-2012",
        "model_type": model_type,
        "vocab": vocab,
        "normalization": norm,
        "normalization_rules": [],
        "add_dummy_prefix": add_dummy,
        "unk_token": unk,
        "bos_token": bos,
        "eos_token": eos,
        "pad_token": pad,
        "merges": [[a, b] for a, b in merges],
        "unigram_pieces": {p: prob for p, prob in uni_pieces},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SentencePiece-lite trainer (mirrors sentencepiece_lite.c train)")
    ap.add_argument("--input", nargs="+", metavar="FILE")
    ap.add_argument("-o", "--output", metavar="FILE")
    ap.add_argument("--model-prefix", default="spm")
    ap.add_argument("--model-type", default="unigram", choices=["bpe", "unigram"])
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--normalization", default="nfkc")
    ap.add_argument("--add-dummy-prefix", dest="dummy", action="store_true", default=True)
    ap.add_argument("--no-add-dummy-prefix", dest="dummy", action="store_false")
    ap.add_argument("--unk-token", default=SP_UNK)
    ap.add_argument("--bos-token", default=SP_BOS)
    ap.add_argument("--eos-token", default=SP_EOS)
    ap.add_argument("--pad-token", default=SP_PAD)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--seed-size", type=int, default=8000)
    ap.add_argument("--max-piece-length", type=int, default=16)
    ap.add_argument("--write-vocab", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    out_path = args.output or f"{args.model_prefix}.model"
    reserved = [args.unk_token, args.bos_token, args.eos_token, args.pad_token]

    line_freq = read_lines(args.input, args.normalization, args.dummy)

    merges = []
    uni_pieces = []

    if args.model_type == "bpe":
        vocab, merges = train_bpe(line_freq, reserved, args.vocab_size, args.min_frequency)
    else:
        vocab, uni_pieces = train_unigram_lite(
            line_freq, reserved, args.vocab_size,
            args.seed_size, args.max_piece_length, args.min_frequency
        )

    write_model(
        out_path, args.model_type, vocab, args.normalization, args.dummy,
        args.unk_token, args.bos_token, args.eos_token, args.pad_token,
        merges, uni_pieces,
    )

    if args.write_vocab:
        vocab_path = f"{args.model_prefix}.vocab"
        with open(vocab_path, "w", encoding="utf-8") as fh:
            for i, tok in enumerate(vocab):
                fh.write(f"{i}\t{tok}\n")

    if args.stats:
        print(
            f'{{"model_type":"{args.model_type}","vocab_size":{len(vocab)},"normalization":"{args.normalization}"}}',
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
