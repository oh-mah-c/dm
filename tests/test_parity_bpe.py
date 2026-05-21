#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.parity_bpe import (
    ParityBPEModel,
    compression_rates,
    evaluate_model,
    initial_sequences,
    learn_parity_bpe,
    load_labeled_tsv,
    read_model,
    write_model,
)


def test_byte_roundtrip() -> None:
    model = ParityBPEModel("none", [(b"a", b"b"), (b"ab", b"c")])
    toks = model.encode_text("abc аб")
    assert model.decode(toks) == "abc аб"


def test_parity_focuses_worst_language() -> None:
    train = {
        "hi": [list(b if isinstance(b, bytes) else bytes([b]) for b in b"abababab")],
        "lo": [list(b if isinstance(b, bytes) else bytes([b]) for b in b"xyzxyz")],
    }
    # Give hi an existing compression advantage, so lo is the min-CR language.
    dev = {
        "hi": [[b"abab", b"abab"]],
        "lo": [[b"x", b"y", b"z", b"x", b"y", b"z"]],
    }
    model, trace = learn_parity_bpe(train, dev, 1, 1, "byte", "none", "parity")
    assert trace[0]["focus_language"] == "lo"
    assert model.merges[0] in [(b"x", b"y"), (b"y", b"z")]


def test_train_evaluate_model_io() -> None:
    train = {
        "en": [sum(initial_sequences("abab abab", "none"), [])],
        "ta": [sum(initial_sequences("வணக்கம் வணக்கம்", "none"), [])],
    }
    model, trace = learn_parity_bpe(train, train, 4, 1, "byte", "none", "parity")
    metrics = evaluate_model(model, train, "byte")
    assert len(model.merges) == 4
    assert metrics["tokens"] > 0
    assert "tokenizer_fairness_gini" in metrics

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pbpe.json"
        write_model(model, path, trace)
        loaded = read_model(path)
        assert loaded.decode(loaded.encode_text("வணக்கம்")) == "வணக்கம்"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == "dm-parity-aware-bpe-2508.04796v2"


def test_labeled_tsv_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy.tsv"
        path.write_text("en\thello\nfr\tbonjour\n", encoding="utf-8")
        corpus = load_labeled_tsv(path, "none")
        rates = compression_rates(corpus, "byte")
        assert sorted(corpus) == ["en", "fr"]
        assert rates["en"] == 1.0


if __name__ == "__main__":
    test_byte_roundtrip()
    test_parity_focuses_worst_language()
    test_train_evaluate_model_io()
    test_labeled_tsv_loader()
    print("parity_bpe tests passed")
