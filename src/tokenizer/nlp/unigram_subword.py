#!/usr/bin/env python3
"""Kudo 2018 unigram LM subword regularization tokenizer.

Implements the core algorithm from:

    Taku Kudo. 2018. Subword Regularization: Improving Neural Network
    Translation Models with Multiple Subword Candidates. ACL 2018,
    paper P18-1007.

The model assumes a sentence segmentation x=(x_1,...,x_M) has probability
prod_i p(x_i). Training alternates EM estimation of subword probabilities with
vocabulary shrinking while always keeping single-character symbols. Inference
supports Viterbi one-best segmentation and stochastic segmentation using
Forward-Filtering Backward-Sampling (FFBS), which is the paper's l=infinity
sampling path for subword regularization.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import random
import sys
from pathlib import Path
from typing import Counter, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

DEFAULT_SEPARATOR = "@@"
LOG_ZERO = -1.0e100


def logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return LOG_ZERO
    m = max(vals)
    if m <= LOG_ZERO / 2:
        return LOG_ZERO
    return m + math.log(sum(math.exp(v - m) for v in vals))


def iter_words(paths: Sequence[Path]) -> Iterator[str]:
    if not paths:
        for line in sys.stdin:
            yield from line.strip().split()
        return
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield from line.strip().split()


def build_word_vocab(paths: Sequence[Path]) -> Counter[str]:
    vocab: Counter[str] = collections.Counter()
    for word in iter_words(paths):
        if word:
            vocab[word] += 1
    return vocab


def enumerate_seed_pieces(
    word_vocab: Mapping[str, int],
    max_piece_length: int,
    max_seed_size: int,
    min_frequency: int,
) -> Counter[str]:
    """Heuristically build the paper's big seed vocabulary.

    The paper allows frequent substring enumeration with enhanced suffix arrays.
    For maintainability and exact word-boundary behavior, this implementation
    enumerates within-word substrings directly and then keeps the most frequent
    pieces. It never creates pieces crossing whitespace boundaries.
    """
    counts: Counter[str] = collections.Counter()
    chars: set[str] = set()
    for word, freq in word_vocab.items():
        chars.update(word)
        n = len(word)
        for i in range(n):
            upper = min(n, i + max_piece_length)
            for j in range(i + 1, upper + 1):
                counts[word[i:j]] += freq
    for ch in chars:
        counts[ch] = max(counts[ch], sum(freq for w, freq in word_vocab.items() if ch in w))
    kept = Counter()
    for piece, freq in counts.most_common():
        if len(piece) == 1 or freq >= min_frequency:
            kept[piece] = freq
        if len(kept) >= max_seed_size:
            break
    for ch in chars:
        kept[ch] = max(kept[ch], 1)
    return kept


def mark_nonfinal(pieces: Sequence[str], separator: str = DEFAULT_SEPARATOR) -> List[str]:
    if not pieces:
        return []
    return [p + separator if i < len(pieces) - 1 else p for i, p in enumerate(pieces)]


def decode_line(line: str, separator: str = DEFAULT_SEPARATOR) -> str:
    text = line.rstrip("\n")
    return text.replace(separator + " ", "").replace(separator, "")


@dataclasses.dataclass
class UnigramModel:
    pieces: Dict[str, float]
    separator: str = DEFAULT_SEPARATOR

    @property
    def log_probs(self) -> Dict[str, float]:
        return {p: math.log(prob) for p, prob in self.pieces.items() if prob > 0.0}

    @property
    def max_piece_length(self) -> int:
        return max((len(p) for p in self.pieces), default=1)

    def lattice(self, word: str, alpha: float = 1.0) -> List[List[Tuple[int, str, float]]]:
        logs = self.log_probs
        n = len(word)
        max_len = self.max_piece_length
        lattice: List[List[Tuple[int, str, float]]] = [[] for _ in range(n + 1)]
        for i in range(n):
            upper = min(n, i + max_len)
            for j in range(i + 1, upper + 1):
                piece = word[i:j]
                if piece in logs:
                    lattice[j].append((i, piece, logs[piece] * alpha))
        return lattice

    def viterbi(self, word: str) -> List[str]:
        n = len(word)
        lattice = self.lattice(word, 1.0)
        best = [LOG_ZERO] * (n + 1)
        prev: List[Optional[Tuple[int, str]]] = [None] * (n + 1)
        best[0] = 0.0
        for end in range(1, n + 1):
            for start, piece, lp in lattice[end]:
                score = best[start] + lp
                if score > best[end]:
                    best[end] = score
                    prev[end] = (start, piece)
        if prev[n] is None:
            return list(word)
        out: List[str] = []
        pos = n
        while pos > 0:
            item = prev[pos]
            if item is None:
                return list(word)
            start, piece = item
            out.append(piece)
            pos = start
        out.reverse()
        return out

    def forward(self, word: str, alpha: float = 1.0) -> Tuple[List[float], List[List[Tuple[int, str, float]]]]:
        n = len(word)
        lattice = self.lattice(word, alpha)
        fwd = [LOG_ZERO] * (n + 1)
        fwd[0] = 0.0
        for end in range(1, n + 1):
            fwd[end] = logsumexp(fwd[start] + lp for start, _, lp in lattice[end])
        return fwd, lattice

    def sample(self, word: str, alpha: float = 1.0, rng: Optional[random.Random] = None) -> List[str]:
        """Sample from all segmentation candidates with FFBS.

        alpha controls smoothness as in the paper: small values flatten the
        distribution; large values approach Viterbi.
        """
        if rng is None:
            rng = random.Random()
        n = len(word)
        fwd, lattice = self.forward(word, alpha)
        if fwd[n] <= LOG_ZERO / 2:
            return list(word)
        out: List[str] = []
        end = n
        while end > 0:
            candidates = lattice[end]
            weights = [math.exp(fwd[start] + lp - fwd[end]) for start, _, lp in candidates]
            total = sum(weights)
            r = rng.random() * total
            acc = 0.0
            chosen = candidates[-1]
            for cand, w in zip(candidates, weights):
                acc += w
                if r <= acc:
                    chosen = cand
                    break
            start, piece, _ = chosen
            out.append(piece)
            end = start
        out.reverse()
        return out

    def nbest(self, word: str, nbest_size: int, alpha: float = 1.0) -> List[Tuple[List[str], float]]:
        beams: List[List[Tuple[float, List[str]]]] = [[] for _ in range(len(word) + 1)]
        beams[0] = [(0.0, [])]
        lattice = self.lattice(word, alpha)
        for end in range(1, len(word) + 1):
            cand: List[Tuple[float, List[str]]] = []
            for start, piece, lp in lattice[end]:
                for score, seq in beams[start]:
                    cand.append((score + lp, seq + [piece]))
            cand.sort(key=lambda x: x[0], reverse=True)
            beams[end] = cand[:nbest_size]
        return beams[len(word)]

    def encode_line(
        self,
        line: str,
        mode: str = "viterbi",
        alpha: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> str:
        encoded: List[str] = []
        for word in line.strip().split():
            if mode == "sample":
                pieces = self.sample(word, alpha, rng)
            else:
                pieces = self.viterbi(word)
            encoded.extend(mark_nonfinal(pieces, self.separator))
        return " ".join(encoded)


def em_step(model: UnigramModel, word_vocab: Mapping[str, int]) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    logs = model.log_probs
    expected: Dict[str, float] = {p: 0.0 for p in model.pieces}
    total_ll = 0.0
    for word, freq in word_vocab.items():
        n = len(word)
        fwd, lattice = model.forward(word, 1.0)
        z = fwd[n]
        if z <= LOG_ZERO / 2:
            for ch in word:
                expected[ch] = expected.get(ch, 0.0) + freq
            total_ll += freq * sum(logs.get(ch, LOG_ZERO) for ch in word)
            continue
        bwd = [LOG_ZERO] * (n + 1)
        bwd[n] = 0.0
        for start in range(n - 1, -1, -1):
            vals = []
            upper = min(n, start + model.max_piece_length)
            for end in range(start + 1, upper + 1):
                piece = word[start:end]
                if piece in logs:
                    vals.append(logs[piece] + bwd[end])
            bwd[start] = logsumexp(vals)
        for end in range(1, n + 1):
            for start, piece, lp in lattice[end]:
                posterior = math.exp(fwd[start] + lp + bwd[end] - z)
                expected[piece] = expected.get(piece, 0.0) + freq * posterior
        total_ll += freq * z
    denom = sum(expected.values())
    if denom <= 0.0:
        denom = 1.0
    new_probs = {p: c / denom for p, c in expected.items() if c > 0.0}
    return new_probs, total_ll, expected


def train_unigram(
    word_vocab: Mapping[str, int],
    vocab_size: int,
    seed_size: int = 8000,
    max_piece_length: int = 16,
    min_frequency: int = 2,
    shrinking_factor: float = 0.8,
    em_sub_iterations: int = 2,
) -> Tuple[UnigramModel, Dict[str, float]]:
    seed = enumerate_seed_pieces(word_vocab, max_piece_length, max(seed_size, vocab_size), min_frequency)
    total = sum(seed.values()) or 1
    model = UnigramModel({p: c / total for p, c in seed.items()})
    chars = {ch for word in word_vocab for ch in word}
    expected: Dict[str, float] = {}
    while len(model.pieces) > vocab_size:
        ll = 0.0
        for _ in range(em_sub_iterations):
            probs, ll, expected = em_step(model, word_vocab)
            for ch in chars:
                probs.setdefault(ch, 1e-12)
            z = sum(probs.values()) or 1.0
            model = UnigramModel({p: v / z for p, v in probs.items()})
        target = max(vocab_size, int(len(model.pieces) * shrinking_factor))
        scored = []
        for piece, prob in model.pieces.items():
            if len(piece) == 1:
                continue
            # Expected contribution to likelihood. Removing high-count/high-prob
            # pieces hurts more, so keep the largest scores.
            score = expected.get(piece, 0.0) * (-math.log(max(prob, 1e-300)))
            scored.append((score, piece))
        scored.sort(reverse=True)
        keep = set(chars)
        nonchar_budget = max(0, target - len(keep))
        keep.update(piece for _, piece in scored[:nonchar_budget])
        if len(keep) >= len(model.pieces):
            break
        kept = {p: v for p, v in model.pieces.items() if p in keep}
        z = sum(kept.values()) or 1.0
        model = UnigramModel({p: v / z for p, v in kept.items()})
    for _ in range(max(1, em_sub_iterations)):
        probs, _, expected = em_step(model, word_vocab)
        for ch in chars:
            probs.setdefault(ch, 1e-12)
        z = sum(probs.values()) or 1.0
        model = UnigramModel({p: v / z for p, v in probs.items()})
    if len(model.pieces) > vocab_size:
        chars = {p for p in model.pieces if len(p) == 1}
        nonchars = sorted(((-math.log(v), p) for p, v in model.pieces.items() if len(p) > 1))
        keep = set(chars)
        keep.update(p for _, p in nonchars[: max(0, vocab_size - len(chars))])
        kept = {p: v for p, v in model.pieces.items() if p in keep}
        z = sum(kept.values()) or 1.0
        model = UnigramModel({p: v / z for p, v in kept.items()})
    return model, expected


def write_model(model: UnigramModel, path: Optional[Path]) -> None:
    out = path.open("w", encoding="utf-8") if path else sys.stdout
    try:
        out.write("#version: dm-unigram-kudo-2018\n")
        for piece, prob in sorted(model.pieces.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"{piece}\t{prob:.17g}\n")
    finally:
        if path:
            out.close()


def read_model(path: Path) -> UnigramModel:
    pieces: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            piece, prob = line.rsplit("\t", 1)
            pieces[piece] = float(prob)
    z = sum(pieces.values()) or 1.0
    return UnigramModel({p: v / z for p, v in pieces.items()})


def command_train(args: argparse.Namespace) -> int:
    vocab = build_word_vocab([Path(p) for p in args.input])
    model, expected = train_unigram(
        vocab,
        vocab_size=args.vocab_size,
        seed_size=args.seed_size,
        max_piece_length=args.max_piece_length,
        min_frequency=args.min_frequency,
        shrinking_factor=args.shrinking_factor,
        em_sub_iterations=args.em_iterations,
    )
    write_model(model, Path(args.output) if args.output else None)
    if args.stats:
        print(json.dumps({
            "word_types": len(vocab),
            "word_tokens": sum(vocab.values()),
            "vocab_size": len(model.pieces),
            "expected_pieces": sum(expected.values()),
        }, ensure_ascii=False), file=sys.stderr)
    return 0


def command_encode(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    rng = random.Random(args.seed)
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            out.write(model.encode_line(line, args.mode, args.alpha, rng))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_nbest(args: argparse.Namespace) -> int:
    model = read_model(Path(args.model))
    for score, pieces in model.nbest(args.word, args.nbest_size, args.alpha):
        print(f"{score:.8f}\t{' '.join(mark_nonfinal(pieces))}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dm_unigram",
        description="Kudo 2018 unigram LM subword regularization tokenizer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train unigram LM subword model")
    train.add_argument("-i", "--input", nargs="+", required=True)
    train.add_argument("-o", "--output", required=True)
    train.add_argument("--vocab-size", type=int, required=True)
    train.add_argument("--seed-size", type=int, default=8000)
    train.add_argument("--max-piece-length", type=int, default=16)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--shrinking-factor", type=float, default=0.8)
    train.add_argument("--em-iterations", type=int, default=2)
    train.add_argument("--stats", action="store_true")
    train.set_defaults(func=command_train)

    enc = sub.add_parser("encode", help="encode corpus with Viterbi or FFBS sampling")
    enc.add_argument("-m", "--model", required=True)
    enc.add_argument("-i", "--input")
    enc.add_argument("-o", "--output")
    enc.add_argument("--mode", choices=["viterbi", "sample"], default="viterbi")
    enc.add_argument("--alpha", type=float, default=1.0)
    enc.add_argument("--seed", type=int, default=None)
    enc.set_defaults(func=command_encode)

    nb = sub.add_parser("nbest", help="print n-best segmentations for one word")
    nb.add_argument("-m", "--model", required=True)
    nb.add_argument("--word", required=True)
    nb.add_argument("-n", "--nbest-size", type=int, default=8)
    nb.add_argument("--alpha", type=float, default=1.0)
    nb.set_defaults(func=command_nbest)

    dec = sub.add_parser("decode", help="remove continuation markers")
    dec.add_argument("-i", "--input")
    dec.add_argument("-o", "--output")
    dec.add_argument("--separator", default=DEFAULT_SEPARATOR)
    dec.set_defaults(func=command_decode)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
