#!/usr/bin/env python3
"""Parity-aware Byte-Pair Encoding from arXiv:2508.04796v2.

Paul et al. modify only BPE vocabulary learning: at every merge step, compute
language-specific compression rates on a labeled development corpus, choose the
currently worst-compressed language, count adjacent token pairs only inside that
language's training split, then apply the chosen merge to all languages. The
resulting tokenizer uses the ordinary deterministic BPE merge procedure at
inference time.
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from typing import Counter, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

Token = bytes
Pair = Tuple[Token, Token]
Corpus = Dict[str, List[List[Token]]]

PRETOKENIZER_PATTERNS = {
    "gpt2": re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+"),
    "gpt4": re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^A-Za-z0-9\s]+|\s+"),
    "whitespace": re.compile(r"\S+"),
}


def b64(token: Token) -> str:
    return base64.b64encode(token).decode("ascii")


def unb64(text: str) -> Token:
    return base64.b64decode(text.encode("ascii"))


def pretokenize(text: str, mode: str) -> List[bytes]:
    if mode == "none":
        raw = text.encode("utf-8")
        return [raw] if raw else []
    if mode not in PRETOKENIZER_PATTERNS:
        raise ValueError(f"unknown pretokenizer: {mode}")
    return [m.group(0).encode("utf-8") for m in PRETOKENIZER_PATTERNS[mode].finditer(text) if m.group(0)]


def initial_sequences(text: str, pretokenizer: str) -> List[List[Token]]:
    return [[bytes([byte]) for byte in chunk] for chunk in pretokenize(text, pretokenizer) if chunk]


def flatten_pretokens(parts: Iterable[List[Token]]) -> List[Token]:
    out: List[Token] = []
    for seq in parts:
        out.extend(seq)
    return out


def parse_lang_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LANG=PATH, got {value!r}")
    lang, path = value.split("=", 1)
    lang = lang.strip()
    if not lang:
        raise ValueError(f"empty language label in {value!r}")
    return lang, Path(path)


def load_lang_corpora(entries: Sequence[str], pretokenizer: str, max_lines: int = 0) -> Corpus:
    corpus: Corpus = {}
    for entry in entries:
        lang, path = parse_lang_path(entry)
        docs = corpus.setdefault(lang, [])
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if max_lines and i >= max_lines:
                    break
                toks = flatten_pretokens(initial_sequences(line.rstrip("\n"), pretokenizer))
                if toks:
                    docs.append(toks)
    return {lang: docs for lang, docs in corpus.items() if docs}


def load_labeled_tsv(path: Path, pretokenizer: str, max_lines: int = 0) -> Corpus:
    corpus: Corpus = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"labeled corpus line {i + 1} must be LANG<TAB>TEXT")
            lang, text = line.rstrip("\n").split("\t", 1)
            toks = flatten_pretokens(initial_sequences(text, pretokenizer))
            if toks:
                corpus.setdefault(lang, []).append(toks)
    return corpus


def merge_sequence(seq: Sequence[Token], pair: Pair) -> List[Token]:
    if len(seq) < 2:
        return list(seq)
    a, b = pair
    repl = a + b
    out: List[Token] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(repl)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def apply_merge(corpus: Corpus, pair: Pair) -> None:
    for docs in corpus.values():
        for i, seq in enumerate(docs):
            docs[i] = merge_sequence(seq, pair)


def pair_counts(seqs: Iterable[Sequence[Token]]) -> Counter[Pair]:
    counts: Counter[Pair] = collections.Counter()
    for seq in seqs:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += 1
    return counts


def count_all_pairs(corpus: Corpus) -> Counter[Pair]:
    counts: Counter[Pair] = collections.Counter()
    for docs in corpus.values():
        counts.update(pair_counts(docs))
    return counts


def original_unit_len(seq: Sequence[Token], unit: str) -> int:
    raw = b"".join(seq)
    if unit == "byte":
        return len(raw)
    text = raw.decode("utf-8", errors="replace")
    if unit == "char":
        return len(text)
    if unit == "line":
        return 1
    raise ValueError(f"unknown CR unit: {unit}")


def compression_rates(corpus: Corpus, unit: str) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for lang, docs in corpus.items():
        vals = []
        for seq in docs:
            if seq:
                vals.append(original_unit_len(seq, unit) / float(len(seq)))
        rates[lang] = sum(vals) / len(vals) if vals else 0.0
    return rates


def gini_cost(rates: Mapping[str, float]) -> float:
    costs = sorted((1.0 / r) for r in rates.values() if r > 0.0)
    n = len(costs)
    total = sum(costs)
    if n == 0 or total == 0.0:
        return 0.0
    weighted = sum((n + 1 - i) * c for i, c in enumerate(costs, start=1))
    return (1.0 / n) * (n + 1 - 2.0 * weighted / total)


def corpus_token_counts(corpus: Corpus) -> Tuple[int, int, Counter[Token]]:
    total_tokens = 0
    total_bytes = 0
    freq: Counter[Token] = collections.Counter()
    for docs in corpus.values():
        for seq in docs:
            total_tokens += len(seq)
            total_bytes += sum(len(tok) for tok in seq)
            freq.update(seq)
    return total_tokens, total_bytes, freq


def renyi_entropy(freq: Mapping[Token, int], alpha: float) -> float:
    total = sum(freq.values())
    if total <= 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-12:
        return -sum((c / total) * math.log2(c / total) for c in freq.values() if c)
    if math.isinf(alpha):
        return -math.log2(max(c / total for c in freq.values()))
    return math.log2(sum((c / total) ** alpha for c in freq.values() if c)) / (1.0 - alpha)


def select_language(
    rates: Mapping[str, float],
    recent: Sequence[str],
    window: int,
    window_limit: int,
) -> str:
    ordered = sorted(rates.items(), key=lambda kv: (kv[1], kv[0]))
    if window <= 0 or window_limit <= 0:
        return ordered[0][0]
    tail = list(recent[-window:])
    for lang, _ in ordered:
        if tail.count(lang) <= window_limit:
            return lang
    return ordered[0][0]


@dataclasses.dataclass
class ParityBPEModel:
    pretokenizer: str
    merges: List[Pair]
    strategy: str = "parity"

    def encode_bytes(self, data: bytes) -> List[Token]:
        seq = [bytes([b]) for b in data]
        for pair in self.merges:
            seq = merge_sequence(seq, pair)
            if len(seq) <= 1:
                break
        return seq

    def encode_text(self, text: str) -> List[Token]:
        out: List[Token] = []
        for chunk in pretokenize(text, self.pretokenizer):
            out.extend(self.encode_bytes(chunk))
        return out

    def decode(self, tokens: Sequence[Token]) -> str:
        return b"".join(tokens).decode("utf-8", errors="replace")

    @property
    def vocab(self) -> set[Token]:
        vocab = {bytes([i]) for i in range(256)}
        for a, b in self.merges:
            vocab.add(a)
            vocab.add(b)
            vocab.add(a + b)
        return vocab


def learn_parity_bpe(
    train: Corpus,
    dev: Corpus,
    num_merges: int,
    min_frequency: int,
    cr_unit: str,
    pretokenizer: str,
    strategy: str,
    hybrid_global_merges: int = 0,
    window: int = 0,
    window_limit: int = 0,
) -> Tuple[ParityBPEModel, List[Dict[str, object]]]:
    if not train:
        raise ValueError("empty training corpus")
    if not dev:
        raise ValueError("empty development corpus")
    train_work: Corpus = {lang: [list(seq) for seq in docs] for lang, docs in train.items()}
    dev_work: Corpus = {lang: [list(seq) for seq in docs] for lang, docs in dev.items()}
    languages = sorted(train_work)
    missing = sorted(set(dev_work) - set(train_work))
    if missing:
        raise ValueError(f"development languages missing in training data: {missing}")

    merges: List[Pair] = []
    trace: List[Dict[str, object]] = []
    recent_langs: List[str] = []

    for k in range(num_merges):
        rates = compression_rates(dev_work, cr_unit)
        use_global = strategy == "classic" or (strategy == "hybrid" and k < hybrid_global_merges)
        if use_global:
            focus = "__global__"
            stats = count_all_pairs(train_work)
        else:
            focus = select_language(rates, recent_langs, window, window_limit)
            stats = pair_counts(train_work.get(focus, []))
            if not stats:
                stats = count_all_pairs(train_work)
                focus = "__global__"

        if not stats:
            break
        best, freq = max(stats.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))
        if freq < min_frequency:
            break
        merges.append(best)
        apply_merge(train_work, best)
        apply_merge(dev_work, best)
        if focus != "__global__":
            recent_langs.append(focus)
        trace.append({
            "step": k + 1,
            "focus_language": focus,
            "pair": [b64(best[0]), b64(best[1])],
            "pair_count": freq,
            "min_cr_before": min(rates.values()) if rates else 0.0,
            "gini_before": gini_cost(rates),
        })

    return ParityBPEModel(pretokenizer=pretokenizer, merges=merges, strategy=strategy), trace


def write_model(model: ParityBPEModel, path: Path, trace: Optional[List[Dict[str, object]]] = None) -> None:
    path.write_text(json.dumps({
        "version": "dm-parity-aware-bpe-2508.04796v2",
        "algorithm": "parity-aware-byte-pair-encoding",
        "pretokenizer": model.pretokenizer,
        "strategy": model.strategy,
        "merges": [[b64(a), b64(b)] for a, b in model.merges],
        "trace": trace or [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def read_model(path: Path) -> ParityBPEModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    merges = [(unb64(a), unb64(b)) for a, b in data["merges"]]
    return ParityBPEModel(data.get("pretokenizer", "none"), merges, data.get("strategy", "parity"))


def load_train_dev(args: argparse.Namespace) -> Tuple[Corpus, Corpus]:
    if args.input_labeled:
        train = load_labeled_tsv(Path(args.input_labeled), args.pretokenizer, args.max_lines)
    else:
        train = load_lang_corpora(args.lang_corpus, args.pretokenizer, args.max_lines)
    if args.dev_labeled:
        dev = load_labeled_tsv(Path(args.dev_labeled), args.pretokenizer, args.max_dev_lines)
    elif args.dev_corpus:
        dev = load_lang_corpora(args.dev_corpus, args.pretokenizer, args.max_dev_lines)
    else:
        dev = {lang: [list(seq) for seq in docs] for lang, docs in train.items()}
    return train, dev


def encode_corpus_as_model(model: ParityBPEModel, entries: Sequence[str], labeled: Optional[str]) -> Corpus:
    if labeled:
        raw = load_labeled_tsv(Path(labeled), model.pretokenizer)
        out: Corpus = {}
        for lang, docs in raw.items():
            out[lang] = [model.encode_bytes(b"".join(seq)) for seq in docs]
        return out
    corpus: Corpus = {}
    for entry in entries:
        lang, path = parse_lang_path(entry)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                toks = model.encode_text(line.rstrip("\n"))
                if toks:
                    corpus.setdefault(lang, []).append(toks)
    return corpus


def evaluate_model(model: ParityBPEModel, corpus: Corpus, cr_unit: str) -> Dict[str, object]:
    rates = compression_rates(corpus, cr_unit)
    tokens, nbytes, freq = corpus_token_counts(corpus)
    vocab = model.vocab
    observed = set(freq)
    return {
        "languages": rates,
        "min_compression_rate": min(rates.values()) if rates else 0.0,
        "mean_compression_rate": sum(rates.values()) / len(rates) if rates else 0.0,
        "tokenizer_fairness_gini": gini_cost(rates),
        "tokens": tokens,
        "utf8_bytes": nbytes,
        "bytes_per_token": (nbytes / tokens) if tokens else 0.0,
        "vocab_size": len(vocab),
        "vocab_observed": len(observed),
        "vocab_utilization": (len(observed) / len(vocab)) if vocab else 0.0,
        "type_token_ratio": (len(observed) / tokens) if tokens else 0.0,
        "renyi_h1": renyi_entropy(freq, 1.0),
        "renyi_h2": renyi_entropy(freq, 2.0),
        "renyi_hinf": renyi_entropy(freq, math.inf),
    }


def command_train(args: argparse.Namespace) -> int:
    train, dev = load_train_dev(args)
    if not train:
        raise SystemExit("no training data loaded")
    strategy = args.strategy
    model, trace = learn_parity_bpe(
        train=train,
        dev=dev,
        num_merges=args.merges,
        min_frequency=args.min_frequency,
        cr_unit=args.cr_unit,
        pretokenizer=args.pretokenizer,
        strategy=strategy,
        hybrid_global_merges=args.hybrid_global_merges,
        window=args.window,
        window_limit=args.window_limit,
    )
    write_model(model, Path(args.output), trace if args.keep_trace else None)
    if args.stats:
        final_dev = encode_corpus_as_model(model, args.dev_corpus or args.lang_corpus, args.dev_labeled or args.input_labeled)
        stats = evaluate_model(model, final_dev, args.cr_unit)
        stats.update({
            "strategy": model.strategy,
            "merges": len(model.merges),
            "languages": stats["languages"],
        })
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
    return 0


def token_to_text(token: Token) -> str:
    text = token.decode("utf-8", errors="replace")
    return text if text else token.hex()


def command_encode(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            toks = model.encode_text(line.rstrip("\n"))
            if args.hex_tokens:
                out.write(json.dumps([tok.hex() for tok in toks], ensure_ascii=False))
            elif args.base64_tokens:
                out.write(json.dumps([b64(tok) for tok in toks], ensure_ascii=False))
            else:
                out.write(" ".join(token_to_text(tok) for tok in toks))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_decode(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            values = json.loads(line)
            if args.base64_tokens:
                toks = [unb64(v) for v in values]
            else:
                toks = [bytes.fromhex(v) for v in values]
            out.write(model.decode(toks))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    corpus = encode_corpus_as_model(model, args.lang_corpus, args.input_labeled)
    print(json.dumps(evaluate_model(model, corpus, args.cr_unit), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_trace(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.model).read_text(encoding="utf-8"))
    trace = data.get("trace", [])
    fields = ["step", "focus_language", "pair_count", "min_cr_before", "gini_before"]
    print(",".join(fields))
    for row in trace:
        print(",".join(str(row.get(f, "")) for f in fields))
    return 0


def add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lang-corpus", action="append", default=[], metavar="LANG=PATH",
                        help="language-labeled corpus file; repeat for multilingual data")
    parser.add_argument("--input-labeled", help="TSV corpus with LANG<TAB>TEXT rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dm_parity_bpe", description="Parity-aware Byte-Pair Encoding from arXiv:2508.04796v2.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="learn classical, parity-aware, hybrid, or window parity BPE merges")
    add_corpus_args(train)
    train.add_argument("--dev-corpus", action="append", default=[], metavar="LANG=PATH")
    train.add_argument("--dev-labeled", help="development TSV with LANG<TAB>TEXT rows")
    train.add_argument("-o", "--output", required=True)
    train.add_argument("--merges", type=int, required=True)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--pretokenizer", choices=["none", "whitespace", "gpt2", "gpt4"], default="none")
    train.add_argument("--cr-unit", choices=["byte", "char", "line"], default="byte")
    train.add_argument("--strategy", choices=["classic", "parity", "hybrid", "window"], default="parity")
    train.add_argument("--hybrid-global-merges", type=int, default=0)
    train.add_argument("--window", type=int, default=0)
    train.add_argument("--window-limit", type=int, default=0)
    train.add_argument("--max-lines", type=int, default=0)
    train.add_argument("--max-dev-lines", type=int, default=0)
    train.add_argument("--keep-trace", action="store_true")
    train.add_argument("--stats", action="store_true")
    train.set_defaults(func=command_train)

    enc = sub.add_parser("encode", help="encode text with learned BPE merges")
    enc.add_argument("-m", "--model", required=True)
    enc.add_argument("-i", "--input")
    enc.add_argument("-o", "--output")
    enc.add_argument("--hex-tokens", action="store_true")
    enc.add_argument("--base64-tokens", action="store_true")
    enc.set_defaults(func=command_encode)

    dec = sub.add_parser("decode", help="decode JSON hex/base64 tokens emitted by encode")
    dec.add_argument("-m", "--model", required=True)
    dec.add_argument("-i", "--input")
    dec.add_argument("-o", "--output")
    dec.add_argument("--base64-tokens", action="store_true")
    dec.set_defaults(func=command_decode)

    ev = sub.add_parser("evaluate", help="compute CR, Gini fairness, vocabulary usage, TTR, and Renyi metrics")
    ev.add_argument("-m", "--model", required=True)
    add_corpus_args(ev)
    ev.add_argument("--cr-unit", choices=["byte", "char", "line"], default="byte")
    ev.set_defaults(func=command_evaluate)

    tr = sub.add_parser("trace", help="print stored merge-focus trace as CSV")
    tr.add_argument("-m", "--model", required=True)
    tr.set_defaults(func=command_trace)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "lang_corpus") and not args.lang_corpus and not getattr(args, "input_labeled", None):
        parser.error("provide --lang-corpus LANG=PATH at least once or --input-labeled TSV")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
