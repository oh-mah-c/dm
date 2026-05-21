#!/usr/bin/env python3
"""Grapheme Pair Encoding (GPE) from COLING 2025 main.400.

The paper argues that tokenizer unfairness for complex scripts often begins
with pre-tokenization and byte/codepoint atomic units. GPE modifies BPE by
extracting grapheme clusters first and using those as the initial vocabulary.
This module implements the runnable algorithm and intrinsic metrics:
Compression Ratio (CR) and Tokenization Parity (TP).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Counter, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

EOW = "</w>"
Pair = Tuple[str, str]

PRETOKENIZER_PATTERNS = {
    "whitespace": re.compile(r"\S+"),
    "gpt2": re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+"),
    "gpt4": re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^A-Za-z0-9\s]+|\s+"),
}

VIRAMAS = {
    "\u094d",  # Devanagari sign virama
    "\u0bcd",  # Tamil sign virama / pulli
    "\u0dca",  # Sinhala sign al-lakuna
}
ZWJ = "\u200d"


def pretokenize(text: str, mode: str) -> List[str]:
    if mode == "none":
        return [text] if text else []
    if mode not in PRETOKENIZER_PATTERNS:
        raise ValueError(f"unknown pretokenizer: {mode}")
    return [m.group(0) for m in PRETOKENIZER_PATTERNS[mode].finditer(text) if m.group(0)]


def is_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def graphemes(text: str) -> List[str]:
    """A deterministic Indic-friendly grapheme extractor.

    Python's stdlib lacks Unicode \\X. This implements the pieces needed by the
    paper's Tamil/Sinhala/Hindi setting: combining marks stay with their base,
    virama/pulli/al-lakuna joins the next consonant, and ZWJ keeps conjuncts
    inside one cluster.
    """
    out: List[str] = []
    cur = ""
    join_next = False
    for ch in text:
        if not cur:
            cur = ch
        elif is_mark(ch) or ch == ZWJ or join_next:
            cur += ch
        else:
            out.append(cur)
            cur = ch
        if ch in VIRAMAS or ch == ZWJ:
            join_next = True
        elif not is_mark(ch):
            join_next = False
    if cur:
        out.append(cur)
    return out


def atomic_units(text: str, unit: str) -> List[str]:
    if unit == "grapheme":
        return graphemes(text)
    if unit == "codepoint":
        return list(text)
    if unit == "byte":
        return [f"{b:02x}" for b in text.encode("utf-8")]
    raise ValueError(f"unknown unit: {unit}")


def original_length(text: str, unit: str) -> int:
    if unit == "byte":
        return len(text.encode("utf-8"))
    if unit == "codepoint":
        return len(text)
    return len(graphemes(text))


def iter_lines(paths: Sequence[Path]) -> Iterator[str]:
    if not paths:
        for line in sys.stdin:
            yield line.rstrip("\n")
        return
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line.rstrip("\n")


def token_word(token: str, unit: str) -> Tuple[str, ...]:
    return tuple(atomic_units(token, unit)) + (EOW,)


def get_pair_stats(vocab: Mapping[Tuple[str, ...], int]) -> Counter[Pair]:
    stats: Counter[Pair] = collections.Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += freq
    return stats


def merge_word(word: Tuple[str, ...], pair: Pair) -> Tuple[str, ...]:
    merged: List[str] = []
    i = 0
    a, b = pair
    repl = a + b
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            merged.append(repl)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def strip_eow(symbols: Sequence[str]) -> List[str]:
    pieces = list(symbols)
    if pieces and pieces[-1] == EOW:
        pieces.pop()
    elif pieces and pieces[-1].endswith(EOW):
        pieces[-1] = pieces[-1][:-len(EOW)]
        if not pieces[-1]:
            pieces.pop()
    return pieces


@dataclasses.dataclass
class GPEModel:
    unit: str
    pretokenizer: str
    merges: List[Pair]

    def encode_pretoken(self, token: str) -> List[str]:
        symbols = list(token_word(token, self.unit))
        for pair in self.merges:
            symbols = list(merge_word(tuple(symbols), pair))
            if len(symbols) == 1:
                break
        return strip_eow(symbols)

    def encode_line(self, line: str) -> List[str]:
        out: List[str] = []
        for tok in pretokenize(line, self.pretokenizer):
            out.extend(self.encode_pretoken(tok))
        return out


def build_vocab(paths: Sequence[Path], unit: str, pretokenizer: str) -> Dict[Tuple[str, ...], int]:
    vocab: Dict[Tuple[str, ...], int] = {}
    for line in iter_lines(paths):
        for tok in pretokenize(line, pretokenizer):
            w = token_word(tok, unit)
            vocab[w] = vocab.get(w, 0) + 1
    return vocab


def learn_gpe(paths: Sequence[Path], unit: str, pretokenizer: str, vocab_size: int, min_frequency: int) -> GPEModel:
    vocab = build_vocab(paths, unit, pretokenizer)
    initial = {sym for word in vocab for sym in word}
    target_merges = max(0, vocab_size - len(initial))
    merges: List[Pair] = []
    for _ in range(target_merges):
        stats = get_pair_stats(vocab)
        if not stats:
            break
        best, freq = max(stats.items(), key=lambda kv: (kv[1], kv[0]))
        if freq < min_frequency:
            break
        merges.append(best)
        vocab = {merge_word(word, best): f for word, f in vocab.items()}
    return GPEModel(unit, pretokenizer, merges)


def write_model(model: GPEModel, path: Path) -> None:
    path.write_text(json.dumps({
        "version": "dm-gpe-coling-2025",
        "unit": model.unit,
        "pretokenizer": model.pretokenizer,
        "merges": model.merges,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def read_model(path: Path) -> GPEModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GPEModel(data["unit"], data["pretokenizer"], [tuple(p) for p in data["merges"]])


def evaluate_model(model: GPEModel, paths: Sequence[Path], length_unit: str) -> Dict[str, float]:
    orig = 0
    toks = 0
    for line in iter_lines(paths):
        orig += original_length(line, length_unit)
        toks += len(model.encode_line(line))
    return {
        "original_length": float(orig),
        "tokenized_length": float(toks),
        "compression_ratio": (float(orig) / float(toks)) if toks else 0.0,
    }


def pretoken_eval(paths: Sequence[Path], pretokenizer: str, length_unit: str) -> Dict[str, float]:
    orig = 0
    toks = 0
    for line in iter_lines(paths):
        orig += original_length(line, length_unit)
        toks += len(pretokenize(line, pretokenizer))
    return {
        "original_length": float(orig),
        "pretoken_count": float(toks),
        "cr_max": (float(orig) / float(toks)) if toks else 0.0,
    }


def tokenization_parity(a_len: float, b_len: float) -> float:
    return a_len / b_len if b_len else 0.0


def command_train(args: argparse.Namespace) -> int:
    model = learn_gpe([Path(p) for p in args.input], args.unit, args.pretokenizer, args.vocab_size, args.min_frequency)
    write_model(model, Path(args.output))
    if args.stats:
        print(json.dumps({
            "unit": model.unit,
            "pretokenizer": model.pretokenizer,
            "merges": len(model.merges),
            "requested_vocab_size": args.vocab_size,
        }, ensure_ascii=False), file=sys.stderr)
    return 0


def command_encode(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            pieces = model.encode_line(line.rstrip("\n"))
            out.write(json.dumps(pieces, ensure_ascii=False) if args.json_tokens else " ".join(pieces))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    print(json.dumps(evaluate_model(model, [Path(p) for p in args.input], args.length_unit), indent=2))
    return 0


def command_pretoken_eval(args: argparse.Namespace) -> int:
    a = pretoken_eval([Path(p) for p in args.input], args.pretokenizer, args.length_unit)
    if args.reference:
        b = pretoken_eval([Path(p) for p in args.reference], args.pretokenizer, args.length_unit)
        a["tokenization_parity_to_reference"] = tokenization_parity(a["pretoken_count"], b["pretoken_count"])
    print(json.dumps(a, indent=2))
    return 0


def command_units(args: argparse.Namespace) -> int:
    for text in args.text:
        print(json.dumps({
            "text": text,
            "bytes": atomic_units(text, "byte"),
            "codepoints": atomic_units(text, "codepoint"),
            "graphemes": atomic_units(text, "grapheme"),
        }, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dm_gpe", description="Grapheme Pair Encoding from COLING 2025 main.400.")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="train GPE/BPE over grapheme, codepoint, or byte atomic units")
    tr.add_argument("-i", "--input", nargs="+", required=True)
    tr.add_argument("-o", "--output", required=True)
    tr.add_argument("--unit", choices=["grapheme", "codepoint", "byte"], default="grapheme")
    tr.add_argument("--pretokenizer", choices=["whitespace", "gpt2", "gpt4", "none"], default="whitespace")
    tr.add_argument("--vocab-size", type=int, required=True)
    tr.add_argument("--min-frequency", type=int, default=2)
    tr.add_argument("--stats", action="store_true")
    tr.set_defaults(func=command_train)

    enc = sub.add_parser("encode", help="encode text with a trained model")
    enc.add_argument("-m", "--model", required=True)
    enc.add_argument("-i", "--input")
    enc.add_argument("-o", "--output")
    enc.add_argument("--json-tokens", action="store_true")
    enc.set_defaults(func=command_encode)

    ev = sub.add_parser("evaluate", help="compute compression ratio")
    ev.add_argument("-m", "--model", required=True)
    ev.add_argument("-i", "--input", nargs="+", required=True)
    ev.add_argument("--length-unit", choices=["grapheme", "codepoint", "byte"], default="grapheme")
    ev.set_defaults(func=command_evaluate)

    pe = sub.add_parser("pretoken-eval", help="compute CRmax and optional TP from pre-tokenization")
    pe.add_argument("-i", "--input", nargs="+", required=True)
    pe.add_argument("--reference", nargs="+")
    pe.add_argument("--pretokenizer", choices=["whitespace", "gpt2", "gpt4", "none"], default="gpt2")
    pe.add_argument("--length-unit", choices=["grapheme", "codepoint", "byte"], default="grapheme")
    pe.set_defaults(func=command_pretoken_eval)

    units = sub.add_parser("units", help="show byte/codepoint/grapheme decomposition")
    units.add_argument("text", nargs="+")
    units.set_defaults(func=command_units)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
