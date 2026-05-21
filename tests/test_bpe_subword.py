#!/usr/bin/env python3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.bpe_subword import decode_line, learn_bpe


def test_paper_toy_example_segments_lower():
    vocab = Counter({"low": 5, "lowest": 2, "newer": 6, "wider": 3})
    model, _ = learn_bpe(vocab, num_merges=10, min_frequency=2)
    assert model.encode_word("lower") == ["low", "er"]


def test_open_vocabulary_word_is_encoded_without_unk():
    vocab = Counter({"low": 5, "lowest": 2, "newer": 6, "wider": 3})
    model, _ = learn_bpe(vocab, num_merges=10, min_frequency=2)
    pieces = model.encode_word("lower")
    assert pieces
    assert "<unk>" not in pieces
    assert "UNK" not in pieces


def test_decode_reverses_continuation_markers():
    assert decode_line("low@@ er newer") == "lower newer"


if __name__ == "__main__":
    test_paper_toy_example_segments_lower()
    test_open_vocabulary_word_is_encoded_without_unk()
    test_decode_reverses_continuation_markers()
    print("bpe_subword tests passed")
