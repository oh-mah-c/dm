#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.sentencepiece_lite import Normalizer, escape_whitespace, read_model, train_model, unescape_whitespace, write_model


class Args:
    pass


def make_args(path: Path, model_type: str, vocab_size: int = 64):
    args = Args()
    args.input = [str(path)]
    args.model_prefix = str(path.with_suffix(""))
    args.output = str(path.with_suffix(".model"))
    args.model_type = model_type
    args.vocab_size = vocab_size
    args.normalization = "nfkc"
    args.normalization_rule_tsv = None
    args.add_dummy_prefix = True
    args.unk_token = "<unk>"
    args.bos_token = "<s>"
    args.eos_token = "</s>"
    args.pad_token = "<pad>"
    args.user_defined_symbol = None
    args.min_frequency = 1
    args.seed_size = 128
    args.max_piece_length = 8
    args.write_vocab = False
    args.stats = False
    return args


def test_escape_is_lossless_with_consecutive_spaces():
    text = "Hello  world."
    escaped = escape_whitespace(text, True)
    assert escaped.startswith("▁")
    assert unescape_whitespace(escaped, True) == text


def test_normalizer_custom_longest_match():
    n = Normalizer("identity", [("abc", "X"), ("ab", "Y")])
    assert n.normalize("abc ab") == "X Y"


def test_bpe_model_roundtrip_raw_text():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "sp.txt"
        corpus.write_text("Hello world.\nこんにちは世界。\nHello  world.\n", encoding="utf-8")
        model = train_model(make_args(corpus, "bpe"))
        pieces = model.encode_pieces("Hello  world.")
        assert model.decode_pieces(pieces) == "Hello  world."
        write_model(model, Path(tmp) / "sp.model")
        loaded = read_model(Path(tmp) / "sp.model")
        assert loaded.decode_ids(loaded.encode_ids("こんにちは世界。")) == "こんにちは世界。"


def test_unigram_model_roundtrip_raw_text():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "sp.txt"
        corpus.write_text("banana bandana\nbanana\n", encoding="utf-8")
        model = train_model(make_args(corpus, "unigram", 40))
        assert model.decode_pieces(model.encode_pieces("banana bandana")) == "banana bandana"
        pieces = [model.bos_token] + model.encode_pieces("banana") + [model.eos_token]
        assert model.decode_pieces(pieces) == "banana"
        ids = [model.token_to_id[model.bos_token]] + model.encode_ids("banana") + [model.token_to_id[model.eos_token]]
        assert model.decode_ids(ids) == "banana"


if __name__ == "__main__":
    test_escape_is_lossless_with_consecutive_spaces()
    test_normalizer_custom_longest_match()
    test_bpe_model_roundtrip_raw_text()
    test_unigram_model_roundtrip_raw_text()
    print("sentencepiece_lite tests passed")
