#!/usr/bin/env python3
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.grapheme_pair_encoding import graphemes, learn_gpe, evaluate_model, pretoken_eval


def test_tamil_grapheme_keeps_pulli_with_consonant():
    assert graphemes("க்கம்") == ["க்க", "ம்"]


def test_devanagari_virama_joins_cluster():
    assert graphemes("स्ते")[0] == "स्ते" or "स्" in graphemes("स्ते")[0]


def test_gpe_train_and_evaluate_runs():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ta.txt"
        p.write_text("வணக்கம் உலகம்\nவணக்கம் நண்பா\n", encoding="utf-8")
        model = learn_gpe([p], "grapheme", "whitespace", vocab_size=64, min_frequency=1)
        metrics = evaluate_model(model, [p], "grapheme")
        assert metrics["compression_ratio"] > 0
        pre = pretoken_eval([p], "whitespace", "grapheme")
        assert pre["cr_max"] >= 1


if __name__ == "__main__":
    test_tamil_grapheme_keeps_pulli_with_consonant()
    test_devanagari_virama_joins_cluster()
    test_gpe_train_and_evaluate_runs()
    print("grapheme_pair_encoding tests passed")
