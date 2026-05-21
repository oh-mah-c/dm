#!/usr/bin/env python3
"""Fast WordPiece tokenization from Song et al., EMNLP 2021.

The implementation follows the paper's LinMaxMatch formulation for single-word
WordPiece tokenization: a vocabulary trie is augmented with failure links and
failure pops, so the greedy longest-match-first segmentation is produced in
linear time in the input length. End-to-end text tokenization preserves the BERT
pre-tokenization behavior of splitting on whitespace and punctuation.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
import unicodedata
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_UNK = "[UNK]"
DEFAULT_SUFFIX = "##"


@dataclasses.dataclass
class TrieNode:
    children: Dict[str, int]
    token: Optional[str] = None
    token_id: Optional[int] = None
    parent: int = -1
    char: str = ""
    text: str = ""
    fail: Optional[int] = None
    pops: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class WordPieceVocab:
    tokens: List[str]
    token_to_id: Dict[str, int]
    unk_token: str = DEFAULT_UNK
    suffix_indicator: str = DEFAULT_SUFFIX

    @classmethod
    def from_file(cls, path: Path, unk_token: str = DEFAULT_UNK, suffix_indicator: str = DEFAULT_SUFFIX) -> "WordPieceVocab":
        tokens: List[str] = []
        seen = set()
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                token = line.split()[0]
                if token in seen:
                    continue
                seen.add(token)
                tokens.append(token)
        if unk_token not in seen:
            tokens.insert(0, unk_token)
        return cls(tokens=tokens, token_to_id={tok: i for i, tok in enumerate(tokens)}, unk_token=unk_token, suffix_indicator=suffix_indicator)


class LinMaxMatchWordPiece:
    def __init__(self, vocab: WordPieceVocab):
        self.vocab = vocab
        self.nodes: List[TrieNode] = [TrieNode(children={}, text="")]
        self.root = 0
        self.suffix_root = self._ensure_path(vocab.suffix_indicator) if vocab.suffix_indicator else self.root
        for token in vocab.tokens:
            self._insert(token, vocab.token_to_id[token])
        self.suffix_root = self._ensure_path(vocab.suffix_indicator) if vocab.suffix_indicator else self.root
        self._precompute_failure()
        self._suffix_indicator_original = self._original_wordpiece(vocab.suffix_indicator)

    def _ensure_path(self, text: str) -> int:
        node = self.root
        for ch in text:
            nxt = self.nodes[node].children.get(ch)
            if nxt is None:
                nxt = len(self.nodes)
                self.nodes[node].children[ch] = nxt
                self.nodes.append(TrieNode(children={}, parent=node, char=ch, text=self.nodes[node].text + ch))
            node = nxt
        return node

    def _insert(self, token: str, token_id: int) -> int:
        node = self._ensure_path(token)
        self.nodes[node].token = token
        self.nodes[node].token_id = token_id
        return node

    def _precompute_failure(self) -> None:
        for node in self.nodes:
            node.fail = None
            node.pops = []
        queue: Deque[int] = collections.deque()
        queue.append(self.root)
        if self.suffix_root != self.root:
            queue.append(self.suffix_root)

        seen_roots = {self.root}
        if self.suffix_root != self.root:
            seen_roots.add(self.suffix_root)

        while queue:
            u = queue.popleft()
            for ch, v in list(self.nodes[u].children.items()):
                if v == self.suffix_root:
                    continue
                vnode = self.nodes[v]
                if vnode.token is not None:
                    vnode.fail = self.suffix_root
                    vnode.pops = [vnode.token]
                else:
                    z = self.nodes[u].fail
                    extra: List[str] = []
                    while z is not None and ch not in self.nodes[z].children:
                        extra.extend(self.nodes[z].pops)
                        z = self.nodes[z].fail
                    if z is not None:
                        vnode.fail = self.nodes[z].children[ch]
                        vnode.pops = list(self.nodes[u].pops) + extra
                queue.append(v)

    def _match_loop(self, text: str, start: int) -> Tuple[List[str], int, int]:
        u = self.root
        tokens: List[str] = []
        i = start
        while i < len(text):
            ch = text[i]
            while ch not in self.nodes[u].children:
                if self.nodes[u].fail is None:
                    return tokens, u, i
                tokens.extend(self.nodes[u].pops)
                u = self.nodes[u].fail
            u = self.nodes[u].children[ch]
            i += 1
        return tokens, u, i

    def tokenize_word(self, word: str) -> List[str]:
        tokens, u, i = self._match_loop(word + " ", 0)
        if i < len(word) or u not in (self.root, self.suffix_root):
            return [self.vocab.unk_token]
        if u == self.suffix_root and not tokens:
            return list(self._suffix_indicator_original)
        return tokens

    def _original_wordpiece(self, word: str) -> List[str]:
        if not word:
            return []
        pieces: List[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            cur = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = self.vocab.suffix_indicator + piece
                if piece in self.vocab.token_to_id:
                    cur = piece
                    break
                end -= 1
            if cur is None:
                return [self.vocab.unk_token]
            pieces.append(cur)
            start = end
        return pieces

    def tokenize_text(self, text: str) -> List[str]:
        result: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and is_space(text[i]):
                i += 1
            if i >= n:
                break
            if is_punctuation(text[i]):
                punct = text[i]
                result.append(punct if punct in self.vocab.token_to_id else self.vocab.unk_token)
                i += 1
                continue
            start = i
            while i < n and not is_space(text[i]) and not is_punctuation(text[i]):
                i += 1
            result.extend(self.tokenize_word(text[start:i]))
        return result

    def ids(self, tokens: Sequence[str]) -> List[int]:
        unk_id = self.vocab.token_to_id.get(self.vocab.unk_token, 0)
        return [self.vocab.token_to_id.get(tok, unk_id) for tok in tokens]


def is_space(ch: str) -> bool:
    return ch.isspace()


def is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def load_tokenizer(model: Path, unk_token: str, suffix_indicator: str) -> LinMaxMatchWordPiece:
    return LinMaxMatchWordPiece(WordPieceVocab.from_file(model, unk_token=unk_token, suffix_indicator=suffix_indicator))


def command_word(args: argparse.Namespace) -> int:
    tok = load_tokenizer(Path(args.vocab), args.unk_token, args.suffix_indicator)
    rows = args.words
    if not rows:
        rows = [line.rstrip("\n") for line in sys.stdin]
    for word in rows:
        pieces = tok.tokenize_word(word)
        print(json.dumps(tok.ids(pieces) if args.ids else pieces, ensure_ascii=False))
    return 0


def command_encode(args: argparse.Namespace) -> int:
    tok = load_tokenizer(Path(args.vocab), args.unk_token, args.suffix_indicator)
    inp = Path(args.input).open("r", encoding="utf-8", errors="replace") if args.input else sys.stdin
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for line in inp:
            pieces = tok.tokenize_text(line.rstrip("\n"))
            if args.ids:
                out.write(json.dumps(tok.ids(pieces), ensure_ascii=False))
            elif args.json_tokens:
                out.write(json.dumps(pieces, ensure_ascii=False))
            else:
                out.write(" ".join(pieces))
            out.write("\n")
    finally:
        if args.input:
            inp.close()
        if args.output:
            out.close()
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    tok = load_tokenizer(Path(args.vocab), args.unk_token, args.suffix_indicator)
    payload = {
        "version": "dm-fast-wordpiece-emnlp-2021",
        "nodes": len(tok.nodes),
        "vocabulary_size": len(tok.vocab.tokens),
        "suffix_indicator": tok.vocab.suffix_indicator,
        "suffix_root": tok.suffix_root,
        "unk_token": tok.vocab.unk_token,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dm_fast_wordpiece", description="LinMaxMatch / E2E WordPiece from EMNLP 2021 main.160.")
    parser.add_argument("--unk-token", default=DEFAULT_UNK)
    parser.add_argument("--suffix-indicator", default=DEFAULT_SUFFIX)
    sub = parser.add_subparsers(dest="command", required=True)

    word = sub.add_parser("word", help="tokenize individual words with LinMaxMatch")
    word.add_argument("-v", "--vocab", required=True)
    word.add_argument("--ids", action="store_true")
    word.add_argument("words", nargs="*")
    word.set_defaults(func=command_word)

    enc = sub.add_parser("encode", help="end-to-end WordPiece tokenization over text")
    enc.add_argument("-v", "--vocab", required=True)
    enc.add_argument("-i", "--input")
    enc.add_argument("-o", "--output")
    enc.add_argument("--json-tokens", action="store_true")
    enc.add_argument("--ids", action="store_true")
    enc.set_defaults(func=command_encode)

    inspect = sub.add_parser("inspect", help="print trie/precomputation summary")
    inspect.add_argument("-v", "--vocab", required=True)
    inspect.set_defaults(func=command_inspect)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
