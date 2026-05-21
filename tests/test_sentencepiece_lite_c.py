#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / "bin" / "dm.exe"


def run(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(DM), *args], cwd=ROOT, text=True, input=input_text, capture_output=True, check=True)


def train_and_roundtrip(model_type: str) -> None:
    subprocess.run(["make", "bin/dm.exe", "bin/dm_sentencepiece"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        corpus = t / "sp.txt"
        corpus.write_text("Hello world.\nこんにちは世界。\nHello  world.\nbanana bandana\n", encoding="utf-8")
        model = t / f"{model_type}.model"
        run(
            "sentencepiece",
            "train",
            "--input",
            str(corpus),
            "--model-type",
            model_type,
            "--vocab-size",
            "80",
            "--min-frequency",
            "1",
            "-o",
            str(model),
        )
        payload = json.loads(model.read_text(encoding="utf-8"))
        assert payload["version"] == "dm-sentencepiece-lite-D18-2012"
        assert payload["model_type"] == model_type

        pieces = run("sentencepiece", "encode", "--model", str(model), "--text", "Hello  world.").stdout
        decoded = run("sentencepiece", "decode", "--model", str(model), input_text=pieces).stdout
        assert decoded == "Hello  world.\n"

        ids = run("sentencepiece", "encode", "--model", str(model), "--text", "banana", "--output-format", "id", "--add-bos", "--add-eos").stdout
        decoded_ids = run("sentencepiece", "decode", "--model", str(model), "--input-format", "id", input_text=ids).stdout
        assert decoded_ids == "banana\n"

        inspect = json.loads(run("sentencepiece", "inspect", "--model", str(model)).stdout)
        assert inspect["vocab_size"] > 4


def test_bpe_model_roundtrip() -> None:
    train_and_roundtrip("bpe")


def test_unigram_model_roundtrip() -> None:
    train_and_roundtrip("unigram")


if __name__ == "__main__":
    test_bpe_model_roundtrip()
    test_unigram_model_roundtrip()
    print("sentencepiece_lite C tests passed")
