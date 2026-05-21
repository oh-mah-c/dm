#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_targets():
    subprocess.run(["make", "bin/dm.exe", "bin/dm_bpe_dropout"], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def write_codes(tmp: Path) -> Path:
    codes = tmp / "codes.bpe"
    codes.write_text("a b\nab c\nabc </w>\n", encoding="utf-8")
    return codes


def run_cmd(args):
    return subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def test_native_bpe_dropout_zero_and_one(tmp: Path):
    build_targets()
    codes = write_codes(tmp)
    text = tmp / "input.txt"
    text.write_text("abc abc\n", encoding="utf-8")

    deterministic = run_cmd([str(ROOT / "bin" / "dm.exe"), "bpe_dropout", "-c", str(codes), "-p", "0", "--seed", "7", "segment", "-i", str(text)])
    assert deterministic == "abc abc\n"

    char_level = run_cmd([str(ROOT / "bin" / "dm.exe"), "bpe_dropout", "-c", str(codes), "-p", "1", "--seed", "7", "segment", "-i", str(text)])
    assert char_level == "a@@ b@@ c a@@ b@@ c\n"


def test_native_standalone_and_stats(tmp: Path):
    build_targets()
    codes = write_codes(tmp)

    sample = run_cmd([str(ROOT / "bin" / "dm_bpe_dropout"), "-c", str(codes), "-p", "0", "--seed", "7", "sample-word", "-n", "2", "abc"]).strip().splitlines()
    assert [json.loads(line) for line in sample] == [
        {"word": "abc", "pieces": ["abc"]},
        {"word": "abc", "pieces": ["abc"]},
    ]

    stats = json.loads(run_cmd([str(ROOT / "bin" / "dm.exe"), "bpe_dropout", "-c", str(codes), "-p", "0.5", "--seed", "7", "stats", "-n", "8", "abc"]))
    assert stats["samples"] == 8
    assert stats["word_tokens"] == 8
    assert stats["distinct_segmentations"] >= 1
    assert stats["avg_pieces_per_word"] >= 1.0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_native_bpe_dropout_zero_and_one(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_native_standalone_and_stats(Path(td))
    print("bpe_dropout_c tests passed")
