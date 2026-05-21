#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_targets():
    subprocess.run(["make", "bin/dm.exe", "bin/dm_fast_wordpiece"], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_cmd(args):
    return subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def test_native_paper_running_example(tmp: Path):
    build_targets()
    vocab = tmp / "vocab.txt"
    vocab.write_text("[UNK]\na\nabcdx\n##b\n##c\n##cdy\n##dz\n", encoding="utf-8")

    out = run_cmd([str(ROOT / "bin" / "dm.exe"), "fast_wordpiece", "word", "-v", str(vocab), "abcdz", "abcz", "abcd"]).strip().splitlines()
    assert [json.loads(line) for line in out] == [["a", "##b", "##c", "##dz"], ["[UNK]"], ["[UNK]"]]


def test_native_wordpiece_suffix_text_and_ids(tmp: Path):
    build_targets()
    vocab = tmp / "vocab.txt"
    vocab.write_text("[UNK]\njohn\njohan\n##son\n'\ns\n", encoding="utf-8")
    text = tmp / "input.txt"
    text.write_text("john johanson's\n", encoding="utf-8")

    encoded = run_cmd([str(ROOT / "bin" / "dm.exe"), "fast_wordpiece", "encode", "-v", str(vocab), "-i", str(text)])
    assert encoded == "john johan ##son ' s\n"

    ids = json.loads(run_cmd([str(ROOT / "bin" / "dm_fast_wordpiece"), "encode", "-v", str(vocab), "-i", str(text), "--ids"]))
    assert ids == [1, 2, 3, 4, 5]

    inspect = json.loads(run_cmd([str(ROOT / "bin" / "dm.exe"), "fast_wordpiece", "inspect", "-v", str(vocab)]))
    assert inspect["version"] == "dm-fast-wordpiece-emnlp-2021"
    assert inspect["vocabulary_size"] == 6
    assert inspect["suffix_indicator"] == "##"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_native_paper_running_example(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_native_wordpiece_suffix_text_and_ids(Path(td))
    print("fast_wordpiece_c tests passed")
