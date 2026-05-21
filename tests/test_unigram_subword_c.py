#!/usr/bin/env python3

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / "bin" / "dm.exe"


def run(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(DM), *args], cwd=ROOT, text=True, input=input_text, capture_output=True, check=True)


def test_unigram_native_train_encode_decode_sample_nbest() -> None:
    subprocess.run(["make", "bin/dm.exe", "bin/dm_unigram"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        corpus = t / "uni.txt"
        corpus.write_text("hello world\nhello hell\nworld word\n", encoding="utf-8")
        model = t / "uni.model"
        run("unigram", "train", "-i", str(corpus), "-o", str(model), "--vocab-size", "16", "--seed-size", "40", "--max-piece-length", "8", "--min-frequency", "1")
        assert model.read_text(encoding="utf-8").startswith("#version: dm-unigram-kudo-2018")

        encoded = run("unigram", "encode", "-m", str(model), "-i", str(corpus), "--mode", "viterbi").stdout
        decoded = run("unigram", "decode", input_text=encoded).stdout
        assert decoded.splitlines()[0] == "hello world"

        sample_a = run("unigram", "encode", "-m", str(model), "-i", str(corpus), "--mode", "sample", "--alpha", "0.5", "--seed", "7").stdout
        sample_b = run("unigram", "encode", "-m", str(model), "-i", str(corpus), "--mode", "sample", "--alpha", "0.5", "--seed", "7").stdout
        assert sample_a == sample_b

        nbest = run("unigram", "nbest", "-m", str(model), "--word", "hello", "-n", "3").stdout
        assert "\t" in nbest


if __name__ == "__main__":
    test_unigram_native_train_encode_decode_sample_nbest()
    print("unigram_subword C tests passed")
