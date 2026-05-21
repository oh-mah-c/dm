#!/usr/bin/env python3
"""BPE-Dropout from Provilkov, Emelianenko, and Voita, ACL 2020.

BPE-Dropout keeps an ordinary BPE merge table but makes segmentation stochastic:
at every segmentation step, each currently possible merge occurrence is removed
with probability p, and the remaining occurrence with the highest BPE priority
is applied. With p=0, this reduces to ordinary BPE segmentation; with p=1,
words remain split into characters.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Counter, Dict, List, Mapping, Optional, Sequence, Tuple

from tokenizer.nlp.bpe_subword import DEFAULT_SEPARATOR, EOW, Pair, mark_nonfinal, read_merges, strip_eow, word_to_symbols

Occurrence = Tuple[int, Pair]


class BPEDropoutSegmenter:
    def __init__(self, merges: Sequence[Pair], dropout: float = 0.1, seed: Optional[int] = None):
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.merges = list(merges)
        self.ranks: Dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self.dropout = dropout
        self.rng = random.Random(seed)

    def possible_merges(self, symbols: Sequence[str]) -> List[Occurrence]:
        out: List[Occurrence] = []
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair in self.ranks:
                out.append((i, pair))
        return out

    def segment_word(self, word: str, dropout: Optional[float] = None) -> List[str]:
        p = self.dropout if dropout is None else dropout
        if p < 0.0 or p > 1.0:
            raise ValueError("dropout must be in [0, 1]")
        symbols = list(word_to_symbols(word))
        while True:
            candidates = []
            for idx, pair in self.possible_merges(symbols):
                if p <= 0.0 or self.rng.random() >= p:
                    candidates.append((idx, pair))
            if not candidates:
                break
            idx, pair = min(candidates, key=lambda item: (self.ranks[item[1]], item[0]))
            symbols = symbols[:idx] + [pair[0] + pair[1]] + symbols[idx + 2:]
        return strip_eow(symbols)

    def segment_line(self, line: str, separator: str = DEFAULT_SEPARATOR, dropout: Optional[float] = None) -> str:
        encoded: List[str] = []
        for word in line.strip().split():
            encoded.extend(mark_nonfinal(self.segment_word(word, dropout), separator))
        return " ".join(encoded)


def segmentation_stats(segmenter: BPEDropoutSegmenter, words: Sequence[str], samples: int, separator: str) -> Dict[str, object]:
    counts: Counter[str] = collections.Counter()
    total_pieces = 0
    total_words = 0
    for _ in range(samples):
        for word in words:
            pieces = segmenter.segment_word(word)
            counts[" ".join(mark_nonfinal(pieces, separator))] += 1
            total_pieces += len(pieces)
            total_words += 1
    return {
        "samples": samples,
        "word_tokens": total_words,
        "distinct_segmentations": len(counts),
        "avg_pieces_per_word": (total_pieces / total_words) if total_words else 0.0,
        "segmentations": dict(counts.most_common()),
    }


def command_segment(args: argparse.Namespace) -> int:
    model = read_merges(Path(args.codes))
    segmenter = BPEDropoutSegmenter(model.merges, args.dropout, args.seed)
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            out.write(segmenter.segment_line(line, args.separator))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_sample_word(args: argparse.Namespace) -> int:
    model = read_merges(Path(args.codes))
    segmenter = BPEDropoutSegmenter(model.merges, args.dropout, args.seed)
    for word in args.words:
        for _ in range(args.samples):
            pieces = mark_nonfinal(segmenter.segment_word(word), args.separator)
            print(json.dumps({"word": word, "pieces": pieces}, ensure_ascii=False))
    return 0


def command_stats(args: argparse.Namespace) -> int:
    model = read_merges(Path(args.codes))
    segmenter = BPEDropoutSegmenter(model.merges, args.dropout, args.seed)
    words: List[str] = []
    if args.input:
        with Path(args.input).open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                words.extend(line.strip().split())
    words.extend(args.words)
    print(json.dumps(segmentation_stats(segmenter, words, args.samples, args.separator), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dm_bpe_dropout", description="BPE-Dropout subword regularization from ACL 2020 main.170.")
    parser.add_argument("-c", "--codes", required=True, help="BPE merge-code file learned by dm bpe learn-bpe")
    parser.add_argument("-p", "--dropout", type=float, default=0.1, help="probability of dropping each possible merge occurrence")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--separator", default=DEFAULT_SEPARATOR)
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="apply stochastic BPE-Dropout segmentation to a corpus")
    seg.add_argument("-i", "--input")
    seg.add_argument("-o", "--output")
    seg.set_defaults(func=command_segment)

    sample = sub.add_parser("sample-word", help="sample multiple segmentations for one or more words")
    sample.add_argument("-n", "--samples", type=int, default=5)
    sample.add_argument("words", nargs="+")
    sample.set_defaults(func=command_sample_word)

    stats = sub.add_parser("stats", help="measure segmentation diversity over words or a corpus")
    stats.add_argument("-i", "--input")
    stats.add_argument("-n", "--samples", type=int, default=20)
    stats.add_argument("words", nargs="*")
    stats.set_defaults(func=command_stats)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
