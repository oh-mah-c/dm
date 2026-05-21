#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.bpe_dropout import BPEDropoutSegmenter, segmentation_stats
from tokenizer.nlp.bpe_subword import learn_bpe, read_merges, write_merges


def test_dropout_zero_matches_bpe_merge_order():
    vocab = {"unrelated": 10}
    model, _ = learn_bpe(vocab, num_merges=20, min_frequency=1)
    det = model.encode_word("unrelated")
    drop = BPEDropoutSegmenter(model.merges, dropout=0.0, seed=1).segment_word("unrelated")
    assert drop == det


def test_dropout_one_returns_characters():
    seg = BPEDropoutSegmenter([("a", "b"), ("ab", "c"), ("c", "</w>")], dropout=1.0, seed=7)
    assert seg.segment_word("abc") == ["a", "b", "c"]


def test_sampling_produces_multiple_segmentations():
    merges = [("u", "n"), ("un", "r"), ("e", "l"), ("a", "t"), ("at", "e"), ("ate", "d</w>")]
    seg = BPEDropoutSegmenter(merges, dropout=0.5, seed=3)
    stats = segmentation_stats(seg, ["unrelated"], samples=25, separator="@@")
    assert stats["distinct_segmentations"] > 1
    assert stats["avg_pieces_per_word"] >= 1.0


def test_codes_file_roundtrip():
    model, _ = learn_bpe({"banana": 3, "bandana": 2}, num_merges=10, min_frequency=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "codes.bpe"
        write_merges(model, path)
        loaded = read_merges(path)
        seg = BPEDropoutSegmenter(loaded.merges, dropout=0.0)
        assert seg.segment_word("banana") == model.encode_word("banana")


if __name__ == "__main__":
    test_dropout_zero_matches_bpe_merge_order()
    test_dropout_one_returns_characters()
    test_sampling_produces_multiple_segmentations()
    test_codes_file_roundtrip()
    print("bpe_dropout tests passed")
