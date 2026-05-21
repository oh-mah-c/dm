#!/usr/bin/env python3
"""Sennrich-Haddow-Birch BPE subword segmentation.

This module implements the word-segmentation algorithm from:

    Rico Sennrich, Barry Haddow, Alexandra Birch. 2016.
    Neural Machine Translation of Rare Words with Subword Units.
    ACL 2016, paper P16-1162.

The implementation follows Algorithm 1 in the paper: extract a word
dictionary with frequencies, represent every word as characters plus an
end-of-word marker, repeatedly merge the most frequent adjacent symbol pair,
and apply learned merges to unseen words at test time.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path
from typing import Counter, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

EOW = "</w>"
DEFAULT_SEPARATOR = "@@"
Pair = Tuple[str, str]
SymbolWord = Tuple[str, ...]


def iter_words(paths: Sequence[Path]) -> Iterator[str]:
    """Yield whitespace-tokenized words from one or more UTF-8 corpora."""
    if not paths:
        for line in sys.stdin:
            yield from line.strip().split()
        return
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield from line.strip().split()


def build_vocabulary(paths: Sequence[Path]) -> Counter[str]:
    vocab: Counter[str] = collections.Counter()
    for word in iter_words(paths):
        if word:
            vocab[word] += 1
    return vocab


def word_to_symbols(word: str) -> SymbolWord:
    """Represent a word as characters plus an explicit end-of-word symbol."""
    return tuple(word) + (EOW,)


def get_pair_stats(vocab: Mapping[SymbolWord, int]) -> Counter[Pair]:
    pairs: Counter[Pair] = collections.Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_word(word: SymbolWord, pair: Pair) -> SymbolWord:
    """Merge every non-overlapping occurrence of pair in one symbol word."""
    if len(word) < 2:
        return word
    merged: List[str] = []
    i = 0
    first, second = pair
    replacement = first + second
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            merged.append(replacement)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def merge_vocab(pair: Pair, vocab: Mapping[SymbolWord, int]) -> Dict[SymbolWord, int]:
    out: Dict[SymbolWord, int] = {}
    for word, freq in vocab.items():
        out[merge_word(word, pair)] = freq
    return out


@dataclasses.dataclass(frozen=True)
class BPEModel:
    merges: Tuple[Pair, ...]
    separator: str = DEFAULT_SEPARATOR

    @property
    def ranks(self) -> Dict[Pair, int]:
        return {pair: i for i, pair in enumerate(self.merges)}

    @property
    def reverse_merges(self) -> Dict[str, Pair]:
        return {a + b: (a, b) for a, b in self.merges}

    def encode_word(
        self,
        word: str,
        vocabulary: Optional[set[str]] = None,
        vocabulary_threshold: int = 1,
        symbol_counts: Optional[Mapping[str, int]] = None,
    ) -> List[str]:
        symbols = list(word_to_symbols(word))
        for pair in self.merges:
            symbols = list(merge_word(tuple(symbols), pair))
            if len(symbols) == 1:
                break
        pieces = strip_eow(symbols)
        if vocabulary is not None:
            pieces = self._repair_unknown_segments(pieces, vocabulary, vocabulary_threshold, symbol_counts or {})
        return pieces

    def encode_line(
        self,
        line: str,
        vocabulary: Optional[set[str]] = None,
        vocabulary_threshold: int = 1,
        symbol_counts: Optional[Mapping[str, int]] = None,
    ) -> str:
        encoded: List[str] = []
        for word in line.strip().split():
            pieces = self.encode_word(word, vocabulary, vocabulary_threshold, symbol_counts)
            encoded.extend(mark_nonfinal(pieces, self.separator))
        return " ".join(encoded)

    def _repair_unknown_segments(
        self,
        pieces: List[str],
        vocabulary: set[str],
        threshold: int,
        symbol_counts: Mapping[str, int],
    ) -> List[str]:
        out: List[str] = []
        reverse = self.reverse_merges
        for piece in pieces:
            out.extend(recursive_split(piece, reverse, vocabulary, threshold, symbol_counts))
        return out


def strip_eow(symbols: Sequence[str]) -> List[str]:
    if not symbols:
        return []
    pieces = list(symbols)
    if pieces[-1] == EOW:
        pieces = pieces[:-1]
    elif pieces[-1].endswith(EOW):
        pieces[-1] = pieces[-1][: -len(EOW)]
        if pieces[-1] == "":
            pieces = pieces[:-1]
    return pieces


def mark_nonfinal(pieces: Sequence[str], separator: str = DEFAULT_SEPARATOR) -> List[str]:
    if not pieces:
        return []
    marked = []
    for i, piece in enumerate(pieces):
        if i < len(pieces) - 1:
            marked.append(piece + separator)
        else:
            marked.append(piece)
    return marked


def decode_line(line: str, separator: str = DEFAULT_SEPARATOR) -> str:
    text = line.rstrip("\n")
    if not text:
        return ""
    return text.replace(separator + " ", "").replace(separator, "")


def learn_bpe(
    vocab: Mapping[str, int],
    num_merges: int,
    min_frequency: int = 2,
) -> Tuple[BPEModel, Counter[str]]:
    """Learn BPE merge operations from a word-frequency dictionary."""
    symbol_vocab: Dict[SymbolWord, int] = {word_to_symbols(word): freq for word, freq in vocab.items()}
    symbol_counts: Counter[str] = collections.Counter()
    for word, freq in symbol_vocab.items():
        for sym in word:
            symbol_counts[sym] += freq

    merges: List[Pair] = []
    for _ in range(num_merges):
        stats = get_pair_stats(symbol_vocab)
        if not stats:
            break
        best, best_freq = max(stats.items(), key=lambda kv: (kv[1], kv[0]))
        if best_freq < min_frequency:
            break
        merges.append(best)
        symbol_vocab = merge_vocab(best, symbol_vocab)
        symbol_counts[best[0] + best[1]] = best_freq
    return BPEModel(tuple(merges)), symbol_counts


def recursive_split(
    piece: str,
    reverse_merges: Mapping[str, Pair],
    vocabulary: set[str],
    threshold: int,
    symbol_counts: Mapping[str, int],
) -> List[str]:
    if piece in vocabulary and symbol_counts.get(piece, threshold) >= threshold:
        return [piece]
    if piece not in reverse_merges:
        return [piece]
    left, right = reverse_merges[piece]
    return (
        recursive_split(left, reverse_merges, vocabulary, threshold, symbol_counts)
        + recursive_split(right, reverse_merges, vocabulary, threshold, symbol_counts)
    )


def read_merges(path: Path) -> BPEModel:
    merges: List[Pair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid BPE merge line in {path}: {line!r}")
            merges.append((parts[0], parts[1]))
    return BPEModel(tuple(merges))


def write_merges(model: BPEModel, path: Optional[Path]) -> None:
    out = path.open("w", encoding="utf-8") if path else sys.stdout
    try:
        out.write("#version: dm-bpe-sennrich-2016\n")
        for a, b in model.merges:
            out.write(f"{a} {b}\n")
    finally:
        if path:
            out.close()


def write_vocab(vocab: Mapping[str, int], path: Optional[Path]) -> None:
    out = path.open("w", encoding="utf-8") if path else sys.stdout
    try:
        for token, freq in sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"{token}\t{freq}\n")
    finally:
        if path:
            out.close()


def read_vocab(path: Path) -> Tuple[set[str], Dict[str, int]]:
    vocab: set[str] = set()
    counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                tok, freq = line.rsplit("\t", 1)
                try:
                    counts[tok] = int(freq)
                except ValueError:
                    counts[tok] = 1
                vocab.add(tok)
            else:
                vocab.add(line)
                counts[line] = 1
    return vocab, counts


def command_learn(args: argparse.Namespace) -> int:
    vocab = build_vocabulary([Path(p) for p in args.input])
    model, symbol_counts = learn_bpe(vocab, args.merges, args.min_frequency)
    write_merges(model, Path(args.output) if args.output else None)
    if args.vocab_out:
        write_vocab(symbol_counts, Path(args.vocab_out))
    if args.stats:
        print(json.dumps({
            "word_types": len(vocab),
            "word_tokens": sum(vocab.values()),
            "merges": len(model.merges),
            "symbol_types": len(symbol_counts),
        }, ensure_ascii=False), file=sys.stderr)
    return 0


def command_apply(args: argparse.Namespace) -> int:
    model = read_merges(Path(args.codes))
    vocab = None
    counts = None
    if args.vocabulary:
        vocab, counts = read_vocab(Path(args.vocabulary))
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            out.write(model.encode_line(line, vocab, args.vocabulary_threshold, counts))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_decode(args: argparse.Namespace) -> int:
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            out.write(decode_line(line, args.separator))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_vocab(args: argparse.Namespace) -> int:
    vocab = build_vocabulary([Path(p) for p in args.input])
    write_vocab(vocab, Path(args.output) if args.output else None)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dm_bpe",
        description="BPE subword segmentation from Sennrich et al. ACL 2016 (P16-1162).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    learn = sub.add_parser("learn-bpe", help="learn BPE merge operations from one or more corpora")
    learn.add_argument("-i", "--input", nargs="+", required=True, help="UTF-8 tokenized corpus paths; pass multiple files for joint BPE")
    learn.add_argument("-o", "--output", help="merge-code output path")
    learn.add_argument("-m", "--merges", type=int, required=True, help="number of BPE merge operations")
    learn.add_argument("--min-frequency", type=int, default=2, help="stop if the best pair is below this frequency")
    learn.add_argument("--vocab-out", help="write learned symbol vocabulary with frequencies")
    learn.add_argument("--stats", action="store_true", help="print JSON statistics to stderr")
    learn.set_defaults(func=command_learn)

    apply = sub.add_parser("apply-bpe", help="apply learned BPE merge operations to a corpus")
    apply.add_argument("-c", "--codes", required=True, help="BPE merge-code file")
    apply.add_argument("-i", "--input", help="input corpus path; stdin if omitted")
    apply.add_argument("-o", "--output", help="output corpus path; stdout if omitted")
    apply.add_argument("--vocabulary", help="optional symbol vocabulary to prevent unknown composed symbols")
    apply.add_argument("--vocabulary-threshold", type=int, default=1, help="minimum allowed vocabulary frequency")
    apply.set_defaults(func=command_apply)

    decode = sub.add_parser("decode", help="remove continuation markers and restore whitespace tokenization")
    decode.add_argument("-i", "--input", help="input path; stdin if omitted")
    decode.add_argument("-o", "--output", help="output path; stdout if omitted")
    decode.add_argument("--separator", default=DEFAULT_SEPARATOR)
    decode.set_defaults(func=command_decode)

    vocab = sub.add_parser("vocab", help="extract word vocabulary frequencies from corpus")
    vocab.add_argument("-i", "--input", nargs="+", required=True)
    vocab.add_argument("-o", "--output")
    vocab.set_defaults(func=command_vocab)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
