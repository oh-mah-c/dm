#!/usr/bin/env python3
"""SentencePiece-style lossless tokenizer/detokenizer from EMNLP 2018 D18-2012.

This module implements the paper's core system design inside dm:
Normalizer -> Trainer -> Encoder -> Decoder, direct training from raw
sentences, whitespace escaping with U+2581, vocabulary/id management, and a
self-contained model file. It supports both subword algorithms discussed in the
paper: BPE and unigram language model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from tokenizer.nlp.bpe_subword import Pair
from tokenizer.nlp.unigram_subword import UnigramModel, train_unigram

SPACE = "\u2581"
DEFAULT_UNK = "<unk>"
DEFAULT_BOS = "<s>"
DEFAULT_EOS = "</s>"
DEFAULT_PAD = "<pad>"


def parse_codepoint_sequence(text: str) -> str:
    out = []
    for part in text.strip().split():
        if part.startswith("U+"):
            out.append(chr(int(part[2:], 16)))
        else:
            out.append(part)
    return "".join(out)


class Normalizer:
    def __init__(self, name: str = "nfkc", rules: Optional[List[Tuple[str, str]]] = None):
        self.name = name
        self.rules = sorted(rules or [], key=lambda x: len(x[0]), reverse=True)

    @classmethod
    def from_tsv(cls, name: str, path: Optional[Path]) -> "Normalizer":
        rules: List[Tuple[str, str]] = []
        if path:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    src, dst = line.split("\t", 1)
                    rules.append((parse_codepoint_sequence(src), parse_codepoint_sequence(dst)))
        return cls(name, rules)

    def normalize(self, text: str) -> str:
        if self.name == "identity":
            out = text
        elif self.name == "lower":
            out = text.lower()
        elif self.name == "nfc":
            out = unicodedata.normalize("NFC", text)
        elif self.name == "nfkc":
            out = unicodedata.normalize("NFKC", text)
        else:
            raise ValueError(f"unknown normalization rule: {self.name}")
        if not self.rules:
            return out
        pieces: List[str] = []
        i = 0
        while i < len(out):
            matched = None
            for src, dst in self.rules:
                if out.startswith(src, i):
                    matched = (src, dst)
                    break
            if matched is None:
                pieces.append(out[i])
                i += 1
            else:
                src, dst = matched
                pieces.append(dst)
                i += len(src)
        return "".join(pieces)


def escape_whitespace(text: str, add_dummy_prefix: bool = True) -> str:
    escaped = text.replace(" ", SPACE)
    return SPACE + escaped if add_dummy_prefix else escaped


def unescape_whitespace(text: str, add_dummy_prefix: bool = True) -> str:
    raw = text.replace(SPACE, " ")
    if add_dummy_prefix and raw.startswith(" "):
        raw = raw[1:]
    return raw


def iter_normalized_lines(paths: Sequence[Path], normalizer: Normalizer, add_dummy_prefix: bool) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield escape_whitespace(normalizer.normalize(line.rstrip("\n")), add_dummy_prefix)


def merge_sequence(seq: Sequence[str], pair: Pair) -> List[str]:
    out: List[str] = []
    i = 0
    a, b = pair
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def bpe_pair_counts(seqs: Mapping[Tuple[str, ...], int]) -> Dict[Pair, int]:
    counts: Dict[Pair, int] = {}
    for seq, freq in seqs.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def train_sentence_bpe(lines: Sequence[str], vocab_size: int, reserved: int, min_frequency: int) -> List[Pair]:
    seqs: Dict[Tuple[str, ...], int] = {}
    for line in lines:
        if line:
            seqs[tuple(line)] = seqs.get(tuple(line), 0) + 1
    chars = {ch for seq in seqs for ch in seq}
    target_merges = max(0, vocab_size - reserved - len(chars))
    merges: List[Pair] = []
    for _ in range(target_merges):
        stats = bpe_pair_counts(seqs)
        if not stats:
            break
        best, freq = max(stats.items(), key=lambda kv: (kv[1], kv[0]))
        if freq < min_frequency:
            break
        merges.append(best)
        seqs = {tuple(merge_sequence(seq, best)): f for seq, f in seqs.items()}
    return merges


@dataclasses.dataclass
class SentencePieceLiteModel:
    model_type: str
    vocab: List[str]
    token_to_id: Dict[str, int]
    normalizer: Normalizer
    add_dummy_prefix: bool = True
    unk_token: str = DEFAULT_UNK
    bos_token: str = DEFAULT_BOS
    eos_token: str = DEFAULT_EOS
    pad_token: str = DEFAULT_PAD
    merges: Optional[List[Pair]] = None
    unigram: Optional[UnigramModel] = None

    def normalize_escape(self, text: str) -> str:
        return escape_whitespace(self.normalizer.normalize(text), self.add_dummy_prefix)

    def encode_pieces(self, text: str, mode: str = "viterbi", alpha: float = 1.0, rng: Optional[random.Random] = None) -> List[str]:
        seq = self.normalize_escape(text)
        if self.model_type == "bpe":
            pieces = list(seq)
            for pair in self.merges or []:
                pieces = merge_sequence(pieces, pair)
            return pieces
        if self.model_type == "unigram":
            if not self.unigram:
                return list(seq)
            return self.unigram.sample(seq, alpha, rng) if mode == "sample" else self.unigram.viterbi(seq)
        raise ValueError(f"unknown model type: {self.model_type}")

    def encode_ids(self, text: str, mode: str = "viterbi", alpha: float = 1.0, rng: Optional[random.Random] = None) -> List[int]:
        unk_id = self.token_to_id.get(self.unk_token, 0)
        return [self.token_to_id.get(piece, unk_id) for piece in self.encode_pieces(text, mode, alpha, rng)]

    def decode_pieces(self, pieces: Sequence[str]) -> str:
        special = {self.unk_token, self.bos_token, self.eos_token, self.pad_token}
        return unescape_whitespace("".join(piece for piece in pieces if piece not in special), self.add_dummy_prefix)

    def decode_ids(self, ids: Sequence[int]) -> str:
        pieces = []
        for idx in ids:
            if 0 <= idx < len(self.vocab):
                tok = self.vocab[idx]
                if tok not in {self.unk_token, self.bos_token, self.eos_token, self.pad_token}:
                    pieces.append(tok)
        return self.decode_pieces(pieces)


def build_vocab(reserved: Sequence[str], pieces: Iterable[str], vocab_size: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in reserved:
        if tok not in seen:
            out.append(tok)
            seen.add(tok)
    for piece in pieces:
        if piece and piece not in seen:
            out.append(piece)
            seen.add(piece)
        if len(out) >= vocab_size:
            break
    return out


def train_model(args: argparse.Namespace) -> SentencePieceLiteModel:
    normalizer = Normalizer.from_tsv(args.normalization, Path(args.normalization_rule_tsv) if args.normalization_rule_tsv else None)
    paths = [Path(p) for p in args.input]
    lines = list(iter_normalized_lines(paths, normalizer, args.add_dummy_prefix))
    reserved = [args.unk_token, args.bos_token, args.eos_token, args.pad_token] + list(args.user_defined_symbol or [])
    if args.model_type == "bpe":
        merges = train_sentence_bpe(lines, args.vocab_size, len(reserved), args.min_frequency)
        pieces = sorted({ch for line in lines for ch in line})
        for a, b in merges:
            pieces.append(a + b)
        vocab = build_vocab(reserved, pieces, args.vocab_size)
        return SentencePieceLiteModel("bpe", vocab, {p: i for i, p in enumerate(vocab)}, normalizer, args.add_dummy_prefix, args.unk_token, args.bos_token, args.eos_token, args.pad_token, merges=merges)
    word_vocab = {line: lines.count(line) for line in set(lines)}
    uni, _ = train_unigram(word_vocab, max(1, args.vocab_size - len(reserved)), args.seed_size, args.max_piece_length, args.min_frequency)
    vocab = build_vocab(reserved, sorted(uni.pieces, key=lambda p: (-uni.pieces[p], p)), args.vocab_size)
    kept = {p: prob for p, prob in uni.pieces.items() if p in set(vocab)}
    z = sum(kept.values()) or 1.0
    uni = UnigramModel({p: v / z for p, v in kept.items()})
    return SentencePieceLiteModel("unigram", vocab, {p: i for i, p in enumerate(vocab)}, normalizer, args.add_dummy_prefix, args.unk_token, args.bos_token, args.eos_token, args.pad_token, unigram=uni)


def write_model(model: SentencePieceLiteModel, path: Path) -> None:
    payload = {
        "version": "dm-sentencepiece-lite-D18-2012",
        "model_type": model.model_type,
        "vocab": model.vocab,
        "normalization": model.normalizer.name,
        "normalization_rules": model.normalizer.rules,
        "add_dummy_prefix": model.add_dummy_prefix,
        "unk_token": model.unk_token,
        "bos_token": model.bos_token,
        "eos_token": model.eos_token,
        "pad_token": model.pad_token,
        "merges": model.merges or [],
        "unigram_pieces": model.unigram.pieces if model.unigram else {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_model(path: Path) -> SentencePieceLiteModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    normalizer = Normalizer(data.get("normalization", "nfkc"), [tuple(x) for x in data.get("normalization_rules", [])])
    vocab = list(data["vocab"])
    uni = UnigramModel({str(k): float(v) for k, v in data.get("unigram_pieces", {}).items()}) if data.get("model_type") == "unigram" else None
    merges = [tuple(p) for p in data.get("merges", [])]
    return SentencePieceLiteModel(
        data["model_type"], vocab, {p: i for i, p in enumerate(vocab)}, normalizer,
        bool(data.get("add_dummy_prefix", True)), data.get("unk_token", DEFAULT_UNK),
        data.get("bos_token", DEFAULT_BOS), data.get("eos_token", DEFAULT_EOS),
        data.get("pad_token", DEFAULT_PAD), merges=merges, unigram=uni)


def command_train(args: argparse.Namespace) -> int:
    model = train_model(args)
    write_model(model, Path(args.model_prefix + ".model" if not args.output else args.output))
    vocab_path = Path(args.model_prefix + ".vocab") if args.write_vocab else None
    if vocab_path:
        vocab_path.write_text("\n".join(f"{i}\t{p}" for i, p in enumerate(model.vocab)) + "\n", encoding="utf-8")
    if args.stats:
        print(json.dumps({"model_type": model.model_type, "vocab_size": len(model.vocab), "normalization": model.normalizer.name}, ensure_ascii=False), file=sys.stderr)
    return 0


def command_encode(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    rng = random.Random(args.seed)
    inp = None if args.text else (Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin)
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        rows = args.text if args.text else (line.rstrip("\n") for line in inp)
        for text in rows:
            pieces = model.encode_pieces(text, args.mode, args.alpha, rng)
            if args.add_bos:
                pieces = [model.bos_token] + pieces
            if args.add_eos:
                pieces = pieces + [model.eos_token]
            if args.output_format == "id":
                unk_id = model.token_to_id.get(model.unk_token, 0)
                out.write(" ".join(str(model.token_to_id.get(piece, unk_id)) for piece in pieces))
            elif args.output_format == "json":
                out.write(json.dumps(pieces, ensure_ascii=False))
            else:
                out.write(" ".join(pieces))
            out.write("\n")
    finally:
        if args.input and inp is not None and inp is not sys.stdin:
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
            line = line.rstrip("\n")
            if args.input_format == "id":
                out.write(model.decode_ids([int(x) for x in line.split() if x]))
            elif args.input_format == "json":
                out.write(model.decode_pieces(json.loads(line)))
            else:
                out.write(model.decode_pieces(line.split()))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    print(json.dumps({
        "version": "dm-sentencepiece-lite-D18-2012",
        "model_type": model.model_type,
        "vocab_size": len(model.vocab),
        "normalization": model.normalizer.name,
        "normalization_rules": len(model.normalizer.rules),
        "add_dummy_prefix": model.add_dummy_prefix,
        "special_tokens": [model.unk_token, model.bos_token, model.eos_token, model.pad_token],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_vocab(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    if args.output_format == "json":
        print(json.dumps([{"id": i, "piece": piece} for i, piece in enumerate(model.vocab)], ensure_ascii=False, indent=2))
    else:
        for i, piece in enumerate(model.vocab):
            print(f"{i}\t{piece}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dm_sentencepiece", description="SentencePiece-style lossless tokenizer/detokenizer from EMNLP 2018 D18-2012.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a self-contained model from raw sentences")
    train.add_argument("--input", nargs="+", required=True)
    train.add_argument("--model-prefix", default="spm")
    train.add_argument("-o", "--output")
    train.add_argument("--model-type", choices=["bpe", "unigram"], default="unigram")
    train.add_argument("--vocab-size", type=int, required=True)
    train.add_argument("--normalization", choices=["nfkc", "nfc", "identity", "lower"], default="nfkc")
    train.add_argument("--normalization-rule-tsv")
    train.add_argument("--add-dummy-prefix", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--unk-token", default=DEFAULT_UNK)
    train.add_argument("--bos-token", default=DEFAULT_BOS)
    train.add_argument("--eos-token", default=DEFAULT_EOS)
    train.add_argument("--pad-token", default=DEFAULT_PAD)
    train.add_argument("--user-defined-symbol", action="append")
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--seed-size", type=int, default=8000)
    train.add_argument("--max-piece-length", type=int, default=16)
    train.add_argument("--write-vocab", action="store_true")
    train.add_argument("--stats", action="store_true")
    train.set_defaults(func=command_train)

    enc = sub.add_parser("encode", help="encode raw text to pieces or ids")
    enc.add_argument("--model", required=True)
    enc.add_argument("--input")
    enc.add_argument("--text", action="append", help="encode this literal string; can be repeated")
    enc.add_argument("--output")
    enc.add_argument("--output-format", choices=["piece", "id", "json"], default="piece")
    enc.add_argument("--mode", choices=["viterbi", "sample"], default="viterbi")
    enc.add_argument("--nbest-size", type=int, default=-1, help="accepted for SentencePiece CLI compatibility; sampling uses the full lattice")
    enc.add_argument("--alpha", type=float, default=1.0)
    enc.add_argument("--seed", type=int)
    enc.add_argument("--add-bos", action="store_true")
    enc.add_argument("--add-eos", action="store_true")
    enc.set_defaults(func=command_encode)

    dec = sub.add_parser("decode", help="decode pieces or ids back to normalized text")
    dec.add_argument("--model", required=True)
    dec.add_argument("--input")
    dec.add_argument("--output")
    dec.add_argument("--input-format", choices=["piece", "id", "json"], default="piece")
    dec.set_defaults(func=command_decode)

    ins = sub.add_parser("inspect", help="print model metadata")
    ins.add_argument("--model", required=True)
    ins.set_defaults(func=command_inspect)

    vocab = sub.add_parser("vocab", help="print model vocabulary ids and pieces")
    vocab.add_argument("--model", required=True)
    vocab.add_argument("--output-format", choices=["tsv", "json"], default="tsv")
    vocab.set_defaults(func=command_vocab)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
