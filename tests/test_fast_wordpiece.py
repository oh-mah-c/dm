#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.fast_wordpiece import LinMaxMatchWordPiece, WordPieceVocab


def make_tokenizer(tokens):
    vocab = WordPieceVocab(tokens=list(tokens), token_to_id={t: i for i, t in enumerate(tokens)})
    return LinMaxMatchWordPiece(vocab)


def test_paper_running_example():
    tok = make_tokenizer(["[UNK]", "a", "abcdx", "##b", "##c", "##cdy", "##dz"])
    assert tok.tokenize_word("abcdz") == ["a", "##b", "##c", "##dz"]
    assert tok.tokenize_word("abcz") == ["[UNK]"]
    assert tok.tokenize_word("abcd") == ["[UNK]"]


def test_wordpiece_suffix_and_text():
    tok = make_tokenizer(["[UNK]", "john", "johan", "##son", "'", "s"])
    assert tok.tokenize_word("johanson") == ["johan", "##son"]
    assert tok.tokenize_text("john johanson's") == ["john", "johan", "##son", "'", "s"]


def test_vocab_file_and_ids():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "vocab.txt"
        p.write_text("[UNK]\nhello\nworld\n!\n", encoding="utf-8")
        vocab = WordPieceVocab.from_file(p)
        tok = LinMaxMatchWordPiece(vocab)
        pieces = tok.tokenize_text("hello world!")
        assert pieces == ["hello", "world", "!"]
        assert tok.ids(pieces) == [1, 2, 3]


if __name__ == "__main__":
    test_paper_running_example()
    test_wordpiece_suffix_and_text()
    test_vocab_file_and_ids()
    print("fast_wordpiece tests passed")
