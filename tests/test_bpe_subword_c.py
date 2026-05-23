#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_targets():
    subprocess.run(["make", "bin/dm.exe", "bin/dm_bpe"], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_cmd(args, input_text=None):
    return subprocess.run(args, cwd=ROOT, input=input_text, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_native_bpe_paper_toy_example(tmp: Path):
    build_targets()
    train = tmp / "train.txt"
    train.write_text("low low low low low lowest lowest newer newer newer newer newer newer wider wider wider\n", encoding="utf-8")
    codes = tmp / "codes.bpe"
    vocab = tmp / "symbols.tsv"

    learned = run_cmd([
        str(ROOT / "bin" / "dm.exe"),
        "bpe",
        "learn-bpe",
        "-i",
        str(train),
        "-m",
        "10",
        "--min-frequency",
        "2",
        "-o",
        str(codes),
        "--vocab-out",
        str(vocab),
        "--stats",
    ])
    # Extract JSON substring from stderr to handle potential logger lines
    stderr_str = learned.stderr
    start_idx = stderr_str.find('{')
    end_idx = stderr_str.rfind('}')
    if start_idx != -1 and end_idx != -1:
        stats = json.loads(stderr_str[start_idx:end_idx+1])
    else:
        stats = json.loads(stderr_str)
    assert stats["word_types"] == 4
    assert stats["word_tokens"] == 16
    assert stats["merges"] == 10

    text = tmp / "input.txt"
    text.write_text("lower newer\n", encoding="utf-8")
    encoded = run_cmd([str(ROOT / "bin" / "dm.exe"), "bpe", "apply-bpe", "-c", str(codes), "-i", str(text)]).stdout
    assert encoded == "low@@ er newer\n"

    decoded = run_cmd([str(ROOT / "bin" / "dm_bpe"), "decode"], input_text="low@@ er newer\n").stdout
    assert decoded == "lower newer\n"


def test_native_bpe_vocab_command(tmp: Path):
    build_targets()
    train = tmp / "train.txt"
    train.write_text("banana banana bandana\n", encoding="utf-8")
    out = run_cmd([str(ROOT / "bin" / "dm_bpe"), "vocab", "-i", str(train)]).stdout.splitlines()
    assert out[:2] == ["banana\t2", "bandana\t1"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_native_bpe_paper_toy_example(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_native_bpe_vocab_command(Path(td))
    print("bpe_subword_c tests passed")
