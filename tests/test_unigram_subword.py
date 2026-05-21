#!/usr/bin/env python3
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.unigram_subword import UnigramModel, decode_line, train_unigram


def test_viterbi_prefers_high_probability_piece_sequence():
    model = UnigramModel({"hello": 0.6, "he": 0.1, "llo": 0.1, "h": 0.05, "e": 0.05, "l": 0.05, "o": 0.05})
    assert model.viterbi("hello") == ["hello"]


def test_ffbs_sampling_is_reproducible_with_seed():
    import random

    model = UnigramModel({"ab": 0.4, "a": 0.3, "b": 0.3})
    a = model.sample("ab", alpha=0.1, rng=random.Random(7))
    b = model.sample("ab", alpha=0.1, rng=random.Random(7))
    assert a == b


def test_training_keeps_characters_and_decodes():
    vocab = Counter({"hello": 5, "hell": 2, "world": 4, "word": 2})
    model, _ = train_unigram(vocab, vocab_size=12, seed_size=40, max_piece_length=8, em_sub_iterations=2)
    chars = set("helloworld")
    assert chars.issubset(set(model.pieces))
    encoded = model.encode_line("hello world")
    assert decode_line(encoded) == "hello world"


if __name__ == "__main__":
    test_viterbi_prefers_high_probability_piece_sequence()
    test_ffbs_sampling_is_reproducible_with_seed()
    test_training_keeps_characters_and_decodes()
    print("unigram_subword tests passed")
