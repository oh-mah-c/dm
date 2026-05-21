"""
Python port of src/tokenizer/parity_bpe.c  train  command.
Implements Parity-Aware BPE: language-aware merge selection that tracks
compression rate per language and focuses merges on the weakest language.

Tokens are byte sequences; merges are base64-encoded in the JSON output.

Usage:
    python train_parity_bpe.py --lang-corpus LANG=FILE [--lang-corpus LANG=FILE ...]
                               --merges K -o model.json
                               [--pretokenizer none|whitespace|gpt2|gpt4]
                               [--strategy parity|classic|hybrid]
                               [--min-frequency N] [--keep-trace] [--stats]
"""

import sys
import math
import json
import base64
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Pretokenizer  (mirrors pretokenize() in parity_bpe.c)
# ---------------------------------------------------------------------------

def _is_alpha(c):
    return c.isascii() and c.isalpha()

def _is_digit(c):
    return c.isascii() and c.isdigit()

def pretokenize(text, mode):
    if mode == "none":
        return [text] if text else []
    if mode == "whitespace":
        return text.split()
    if mode not in ("gpt2", "gpt4"):
        raise ValueError(f"Unknown pretokenizer: {mode}")

    gpt4 = mode == "gpt4"
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        # contractions starting with '
        if text[i] == "'" and i + 1 < n:
            matched = False
            for tail in ("s", "t", "re", "ve", "m", "ll", "d"):
                if text[i + 1: i + 1 + len(tail)] == tail:
                    tokens.append(text[i: i + 1 + len(tail)])
                    i += 1 + len(tail)
                    matched = True
                    break
            if matched:
                continue

        # leading space before non-space
        start = i
        if text[i] == " " and i + 1 < n and text[i + 1] != " ":
            i += 1

        if i < n and _is_alpha(text[i]):
            while i < n and _is_alpha(text[i]):
                i += 1
        elif i < n and _is_digit(text[i]):
            if gpt4:
                limit = min(i + 3, n)
            else:
                limit = n
            while i < limit and _is_digit(text[i]):
                i += 1
        elif i < n and not text[i].isspace():
            while i < n and not text[i].isspace() and not _is_alpha(text[i]) and not _is_digit(text[i]):
                i += 1
        else:
            while i < n and text[i].isspace():
                i += 1

        if i > start:
            tokens.append(text[start:i])
    return tokens


def text_to_bytes(text, pretok):
    """Convert text to list of single-byte values (list of bytes objects of length 1)."""
    chunks = pretokenize(text, pretok)
    result = []
    for chunk in chunks:
        for byte in chunk.encode("utf-8"):
            result.append(bytes([byte]))
    return result


# ---------------------------------------------------------------------------
# Corpus I/O
# ---------------------------------------------------------------------------

def parse_lang_path(entry):
    eq = entry.index("=")
    return entry[:eq], entry[eq + 1:]


def load_lang_corpora(entries, pretok, max_lines=0):
    """Returns {lang: [[bytes, ...], ...]} — docs are lists of single-byte tokens."""
    corpus = {}
    for entry in entries:
        lang, path = parse_lang_path(entry)
        docs = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh):
                if max_lines and n >= max_lines:
                    break
                line = line.rstrip("\n\r")
                seq = text_to_bytes(line, pretok)
                if seq:
                    docs.append(seq)
        corpus[lang] = docs
    return corpus


def load_labeled_tsv(path, pretok, max_lines=0):
    corpus = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            if max_lines and n >= max_lines:
                break
            line = line.rstrip("\n\r")
            if not line:
                continue
            tab = line.index("\t")
            lang = line[:tab]
            text = line[tab + 1:]
            seq = text_to_bytes(text, pretok)
            if seq:
                corpus.setdefault(lang, []).append(seq)
    return corpus


def corpus_copy(corpus):
    return {lang: [list(doc) for doc in docs] for lang, docs in corpus.items()}


# ---------------------------------------------------------------------------
# Pair statistics
# ---------------------------------------------------------------------------

def collect_pairs_docs(docs):
    stats = defaultdict(int)
    for doc in docs:
        for i in range(len(doc) - 1):
            stats[(doc[i], doc[i + 1])] += 1
    return stats


def collect_all_pairs(corpus):
    stats = defaultdict(int)
    for docs in corpus.values():
        for doc in docs:
            for i in range(len(doc) - 1):
                stats[(doc[i], doc[i + 1])] += 1
    return stats


def best_pair(stats, min_freq):
    if not stats:
        return None
    best = max(stats, key=lambda p: (stats[p], p))
    return best if stats[best] >= min_freq else None


# ---------------------------------------------------------------------------
# Merge application
# ---------------------------------------------------------------------------

def apply_merge_doc(doc, left, right):
    joined = left + right
    result = []
    i = 0
    while i < len(doc):
        if i + 1 < len(doc) and doc[i] == left and doc[i + 1] == right:
            result.append(joined)
            i += 2
        else:
            result.append(doc[i])
            i += 1
    return result


def apply_merge_corpus(corpus, left, right):
    for lang in corpus:
        corpus[lang] = [apply_merge_doc(doc, left, right) for doc in corpus[lang]]


# ---------------------------------------------------------------------------
# Compression rate and Gini  (mirrors compression_rate_docs / gini_from_rates)
# ---------------------------------------------------------------------------

def seq_original_len(doc, unit):
    if unit == "byte":
        return sum(len(t) for t in doc)
    if unit == "line":
        return 1.0
    if unit == "char":
        total = 0
        for t in doc:
            for byte in t:
                if (byte & 0xC0) != 0x80:
                    total += 1
        return float(total)
    return sum(len(t) for t in doc)


def compression_rate(docs, unit):
    rates = []
    for doc in docs:
        if doc:
            orig = seq_original_len(doc, unit)
            rates.append(orig / len(doc))
    return sum(rates) / len(rates) if rates else 0.0


def gini_from_rates(rates):
    if not rates:
        return 0.0
    costs = sorted(1.0 / r for r in rates if r > 0.0)
    m = len(costs)
    if m == 0:
        return 0.0
    total = sum(costs)
    weighted = sum((m - i) * c for i, c in enumerate(costs))
    return (1.0 / m) * (m + 1.0 - 2.0 * weighted / total) if total else 0.0


def select_language(dev, rates, recent, window, window_limit):
    langs = list(dev.keys())
    best = min(langs, key=lambda l: (rates[l], l))
    if not window or not window_limit:
        return best
    for _ in range(len(langs)):
        for lang in sorted(langs, key=lambda l: (rates[l], l)):
            start = max(len(recent) - window, 0)
            cnt = sum(1 for r in recent[start:] if r == lang)
            if cnt <= window_limit:
                return lang
    return best


# ---------------------------------------------------------------------------
# Core training loop  (mirrors learn_pbpe)
# ---------------------------------------------------------------------------

def learn_pbpe(train, dev, num_merges, min_freq, unit, pretok,
               strategy, hybrid_global, window, window_limit):
    tw = corpus_copy(train)
    dw = corpus_copy(dev)
    merges = []
    trace = []
    recent = []

    for k in range(num_merges):
        rates = {lang: compression_rate(dw[lang], unit) for lang in dw}
        min_cr = min(rates.values()) if rates else 0.0
        gini = gini_from_rates(list(rates.values()))

        is_global = strategy == "classic" or (strategy == "hybrid" and k < hybrid_global)
        focus = "__global__" if is_global else select_language(dw, rates, recent, window, window_limit)

        if is_global:
            stats = collect_all_pairs(tw)
        else:
            lang_docs = tw.get(focus)
            if lang_docs:
                stats = collect_pairs_docs(lang_docs)
            else:
                stats = defaultdict(int)
            if not stats:
                stats = collect_all_pairs(tw)
                focus = "__global__"

        pair = best_pair(stats, min_freq)
        if pair is None:
            break

        left, right = pair
        merges.append((left, right))
        apply_merge_corpus(tw, left, right)
        apply_merge_corpus(dw, left, right)
        trace.append({
            "step": k + 1,
            "focus_language": focus,
            "pair": [left, right],
            "pair_count": stats[pair],
            "min_cr_before": min_cr,
            "gini_before": gini,
        })

        if focus != "__global__":
            recent.append(focus)

    return merges, trace


# ---------------------------------------------------------------------------
# JSON model writer  (mirrors write_model in parity_bpe.c)
# ---------------------------------------------------------------------------

def b64(tok):
    return base64.b64encode(tok).decode("ascii")


def write_model(path, pretok, strategy, merges, trace, keep_trace):
    model = {
        "version": "dm-parity-aware-bpe-2508.04796v2",
        "algorithm": "parity-aware-byte-pair-encoding",
        "pretokenizer": pretok,
        "strategy": strategy,
        "merges": [[b64(l), b64(r)] for l, r in merges],
        "trace": [] if not keep_trace else [
            {
                "step": row["step"],
                "focus_language": row["focus_language"],
                "pair": [b64(row["pair"][0]), b64(row["pair"][1])],
                "pair_count": row["pair_count"],
                "min_cr_before": row["min_cr_before"],
                "gini_before": row["gini_before"],
            }
            for row in trace
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parity-Aware BPE trainer (mirrors parity_bpe.c train)")
    ap.add_argument("--lang-corpus", action="append", metavar="LANG=FILE", default=[])
    ap.add_argument("--dev-corpus", action="append", metavar="LANG=FILE", default=[])
    ap.add_argument("--input-labeled", metavar="TSV")
    ap.add_argument("--dev-labeled", metavar="TSV")
    ap.add_argument("-o", "--output", required=True, metavar="FILE")
    ap.add_argument("--merges", type=int, required=True)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--pretokenizer", default="none",
                    choices=["none", "whitespace", "gpt2", "gpt4"])
    ap.add_argument("--cr-unit", default="byte", choices=["byte", "char", "line"])
    ap.add_argument("--strategy", default="parity",
                    choices=["parity", "classic", "hybrid"])
    ap.add_argument("--hybrid-global-merges", type=int, default=0)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--window-limit", type=int, default=0)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--max-dev-lines", type=int, default=0)
    ap.add_argument("--keep-trace", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if not args.output or not args.merges or (not args.input_labeled and not args.lang_corpus):
        ap.print_help()
        sys.exit(2)

    if args.input_labeled:
        train = load_labeled_tsv(args.input_labeled, args.pretokenizer, args.max_lines)
    else:
        train = load_lang_corpora(args.lang_corpus, args.pretokenizer, args.max_lines)

    if args.dev_labeled:
        dev = load_labeled_tsv(args.dev_labeled, args.pretokenizer, args.max_dev_lines)
    elif args.dev_corpus:
        dev = load_lang_corpora(args.dev_corpus, args.pretokenizer, args.max_dev_lines)
    else:
        dev = corpus_copy(train)

    merges, trace = learn_pbpe(
        train, dev,
        args.merges, args.min_frequency,
        args.cr_unit, args.pretokenizer,
        args.strategy, args.hybrid_global_merges,
        args.window, args.window_limit,
    )

    write_model(args.output, args.pretokenizer, args.strategy, merges, trace, args.keep_trace)

    if args.stats:
        total_docs = sum(len(docs) for docs in train.values())
        print(
            f'{{"merges":{len(merges)},"langs":{list(train.keys())},"docs":{total_docs}}}',
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
