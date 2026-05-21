#!/usr/bin/env python3
"""Tokenizer design lab for Dagan et al. 2024 (arXiv:2402.01035v2).

The paper studies practical levers for LLM tokenizer adaptation:
training data mix, BPE vocabulary size, pre-tokenization regular expression,
compression metrics such as Normalized Sequence Length (NSL), and inference /
memory trade-offs. This module implements those runnable components for dm.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from typing import Counter, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from tokenizer.nlp.bpe_subword import get_pair_stats, merge_vocab, merge_word, word_to_symbols, strip_eow

Pair = Tuple[str, str]


PRETOKENIZER_PATTERNS = {
    # GPT-2 style approximation: optional leading space, letters, unbounded
    # digits, punctuation groups, and whitespace groups.
    "gpt2": re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+"),
    # GPT-4 style approximation: same spirit, but digit runs are capped to 3.
    "gpt4": re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^A-Za-z0-9\s]+|\s+"),
    # Punct removes English contractions and avoids punctuation/whitespace being
    # absorbed into alpha-only tokens. This favors compositional code tokens.
    "punct": re.compile(r"[A-Za-z]+|[0-9]{1,3}|[^\S\r\n]+|\r?\n|[^A-Za-z0-9\s]"),
}


def pretokenize(text: str, mode: str) -> List[str]:
    if mode == "identity":
        return [text] if text else []
    if mode not in PRETOKENIZER_PATTERNS:
        raise ValueError(f"unknown pretokenizer: {mode}")
    return [m.group(0) for m in PRETOKENIZER_PATTERNS[mode].finditer(text) if m.group(0)]


def iter_text(paths: Sequence[Path], max_chars: int = 0) -> Iterator[str]:
    remaining = max_chars
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if max_chars > 0:
                    if remaining <= 0:
                        return
                    chunk = line[:remaining]
                    remaining -= len(chunk)
                    yield chunk
                else:
                    yield line


def build_chunk_vocab(paths: Sequence[Path], pretokenizer: str, max_chars: int = 0) -> Counter[str]:
    vocab: Counter[str] = collections.Counter()
    for text in iter_text(paths, max_chars):
        for chunk in pretokenize(text, pretokenizer):
            vocab[chunk] += 1
    return vocab


def learn_bpe_from_chunks(
    chunks: Mapping[str, int],
    vocab_size: int,
    min_frequency: int = 2,
) -> List[Pair]:
    symbol_vocab = {word_to_symbols(chunk): freq for chunk, freq in chunks.items()}
    chars = {sym for word in symbol_vocab for sym in word}
    target_merges = max(0, vocab_size - len(chars))
    merges: List[Pair] = []
    for _ in range(target_merges):
        stats = get_pair_stats(symbol_vocab)
        if not stats:
            break
        best, freq = max(stats.items(), key=lambda kv: (kv[1], kv[0]))
        if freq < min_frequency:
            break
        merges.append(best)
        symbol_vocab = merge_vocab(best, symbol_vocab)
    return merges


@dataclasses.dataclass
class LabBPEModel:
    pretokenizer: str
    merges: List[Pair]

    def encode_chunk(self, chunk: str) -> List[str]:
        symbols = list(word_to_symbols(chunk))
        for pair in self.merges:
            symbols = list(merge_word(tuple(symbols), pair))
            if len(symbols) == 1:
                break
        return strip_eow(symbols)

    def encode_text(self, text: str) -> List[str]:
        out: List[str] = []
        for chunk in pretokenize(text, self.pretokenizer):
            out.extend(self.encode_chunk(chunk))
        return out

    @property
    def vocab(self) -> set[str]:
        v = set()
        for a, b in self.merges:
            v.add(a)
            v.add(b)
            v.add(a + b)
        return v


def write_model(model: LabBPEModel, path: Path) -> None:
    payload = {
        "version": "dm-tokenizer-lab-dagan-2024",
        "algorithm": "bpe",
        "pretokenizer": model.pretokenizer,
        "merges": model.merges,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_model(path: Path) -> LabBPEModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LabBPEModel(payload["pretokenizer"], [tuple(x) for x in payload["merges"]])


def count_tokens(model: LabBPEModel, paths: Sequence[Path]) -> Tuple[int, int, Counter[str]]:
    total_tokens = 0
    total_bytes = 0
    freq: Counter[str] = collections.Counter()
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for text in fh:
                toks = model.encode_text(text)
                total_tokens += len(toks)
                total_bytes += len(text.encode("utf-8"))
                freq.update(toks)
    return total_tokens, total_bytes, freq


def renyi_entropy(freq: Mapping[str, int], alpha: float = 2.5) -> float:
    total = sum(freq.values())
    if total <= 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-12:
        return -sum((c / total) * math.log(c / total) for c in freq.values() if c)
    s = sum((c / total) ** alpha for c in freq.values() if c)
    return math.log(s) / (1.0 - alpha)


def evaluate_model(model: LabBPEModel, paths: Sequence[Path], baseline: Optional[LabBPEModel]) -> Dict[str, float]:
    toks, nbytes, freq = count_tokens(model, paths)
    baseline_tokens = None
    if baseline is not None:
        baseline_tokens, _, _ = count_tokens(baseline, paths)
    return {
        "tokens": float(toks),
        "utf8_bytes": float(nbytes),
        "bytes_per_token": (float(nbytes) / float(toks)) if toks else 0.0,
        "nsl": (float(toks) / float(baseline_tokens)) if baseline_tokens else 1.0,
        "vocab_observed": float(len(freq)),
        "renyi_alpha_2p5": renyi_entropy(freq, 2.5),
    }


def vocabulary_memory_params(vocab_size: int, dim: int) -> int:
    return 2 * dim * vocab_size


def cache_params(sequence_len: float, layers: int, batch: int, dim: int, kv_heads: int, heads: int) -> float:
    return 2.0 * layers * batch * dim * (float(kv_heads) / float(heads)) * sequence_len


def command_train(args: argparse.Namespace) -> int:
    chunks = build_chunk_vocab([Path(p) for p in args.input], args.pretokenizer, args.max_chars)
    merges = learn_bpe_from_chunks(chunks, args.vocab_size, args.min_frequency)
    model = LabBPEModel(args.pretokenizer, merges)
    write_model(model, Path(args.output))
    if args.stats:
        print(json.dumps({
            "chunk_types": len(chunks),
            "chunk_tokens": sum(chunks.values()),
            "pretokenizer": args.pretokenizer,
            "requested_vocab_size": args.vocab_size,
            "merges": len(merges),
        }, ensure_ascii=False), file=sys.stderr)
    return 0


def command_encode(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            toks = model.encode_text(line)
            if args.json_tokens:
                out.write(json.dumps(toks, ensure_ascii=False))
            else:
                out.write(" ".join(toks))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_eval(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    baseline = read_model(Path(args.baseline)) if args.baseline else None
    metrics = evaluate_model(model, [Path(p) for p in args.input], baseline)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    baseline = read_model(Path(args.baseline)) if args.baseline else None
    rows = []
    for path in args.models:
        model = read_model(Path(path))
        metrics = evaluate_model(model, [Path(p) for p in args.input], baseline)
        metrics["model"] = path
        metrics["pretokenizer"] = model.pretokenizer
        rows.append(metrics)
    fields = ["model", "pretokenizer", "tokens", "utf8_bytes", "bytes_per_token", "nsl", "vocab_observed", "renyi_alpha_2p5"]
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[f]) for f in fields))
    return 0


def command_vocab_tradeoff(args: argparse.Namespace) -> int:
    entries = []
    with Path(args.nsl_csv).open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        idx_v = header.index("vocab_size")
        idx_nsl = header.index("nsl32k")
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            v = int(float(parts[idx_v]))
            nsl = float(parts[idx_nsl])
            seq = args.sequence_len_32k * nsl
            vocab_mem = vocabulary_memory_params(v, args.dim)
            cache = cache_params(seq, args.layers, args.batch, args.dim, args.kv_heads, args.heads)
            total = vocab_mem + cache
            inf_cost = nsl * (1.0 + args.softmax_slope * (v - 32000) / 32000.0)
            entries.append((total, inf_cost, v, nsl, vocab_mem, cache))
    best_mem = min(entries, key=lambda x: x[0])
    best_inf = min(entries, key=lambda x: x[1])
    print(json.dumps({
        "memory_optimal_vocab_size": best_mem[2],
        "inference_optimal_vocab_size": best_inf[2],
        "memory_optimal": {
            "vocab_size": best_mem[2],
            "nsl32k": best_mem[3],
            "vocab_params": best_mem[4],
            "cache_params": best_mem[5],
            "total_params_proxy": best_mem[0],
        },
        "inference_optimal": {
            "vocab_size": best_inf[2],
            "nsl32k": best_inf[3],
            "cost_proxy": best_inf[1],
        },
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dm_tokenizer_lab", description="BPE tokenizer design lab from arXiv:2402.01035v2.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train-bpe", help="train BPE tokenizer with a selected pre-tokenizer")
    train.add_argument("-i", "--input", nargs="+", required=True)
    train.add_argument("-o", "--output", required=True)
    train.add_argument("--vocab-size", type=int, required=True)
    train.add_argument("--pretokenizer", choices=["gpt2", "gpt4", "punct", "identity"], default="gpt4")
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--max-chars", type=int, default=0)
    train.add_argument("--stats", action="store_true")
    train.set_defaults(func=command_train)

    enc = sub.add_parser("encode", help="encode text with a trained tokenizer-lab model")
    enc.add_argument("-m", "--model", required=True)
    enc.add_argument("-i", "--input")
    enc.add_argument("-o", "--output")
    enc.add_argument("--json-tokens", action="store_true")
    enc.set_defaults(func=command_encode)

    ev = sub.add_parser("evaluate", help="compute compression metrics")
    ev.add_argument("-m", "--model", required=True)
    ev.add_argument("-i", "--input", nargs="+", required=True)
    ev.add_argument("--baseline")
    ev.set_defaults(func=command_eval)

    comp = sub.add_parser("compare", help="CSV comparison for multiple models")
    comp.add_argument("-m", "--models", nargs="+", required=True)
    comp.add_argument("-i", "--input", nargs="+", required=True)
    comp.add_argument("--baseline")
    comp.set_defaults(func=command_compare)

    trade = sub.add_parser("vocab-tradeoff", help="estimate memory/inference optimal vocabulary size from nsl CSV")
    trade.add_argument("--nsl-csv", required=True, help="CSV with columns vocab_size,nsl32k")
    trade.add_argument("--dim", type=int, required=True)
    trade.add_argument("--layers", type=int, required=True)
    trade.add_argument("--heads", type=int, required=True)
    trade.add_argument("--kv-heads", type=int, required=True)
    trade.add_argument("--batch", type=int, default=1)
    trade.add_argument("--sequence-len-32k", type=float, default=4096.0)
    trade.add_argument("--softmax-slope", type=float, default=0.04)
    trade.set_defaults(func=command_vocab_tradeoff)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
