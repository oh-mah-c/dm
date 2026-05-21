#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_targets():
    subprocess.run(["make", "bin/dm.exe", "bin/dm_gpe"], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_cmd(args):
    return subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_native_units_keep_indic_graphemes():
    build_targets()
    rows = run_cmd([str(ROOT / "bin" / "dm.exe"), "gpe", "units", "க்கம்", "स्ते"]).stdout.strip().splitlines()
    tamil = json.loads(rows[0])
    devanagari = json.loads(rows[1])
    assert tamil["graphemes"] == ["க்க", "ம்"]
    assert devanagari["graphemes"][0].startswith("स्")


def test_native_gpe_train_encode_evaluate(tmp: Path):
    build_targets()
    corpus = tmp / "ta.txt"
    corpus.write_text("வணக்கம் உலகம்\nவணக்கம் நண்பா\n", encoding="utf-8")
    model = tmp / "gpe.json"

    trained = run_cmd([
        str(ROOT / "bin" / "dm.exe"),
        "gpe",
        "train",
        "-i",
        str(corpus),
        "--unit",
        "grapheme",
        "--pretokenizer",
        "whitespace",
        "--vocab-size",
        "64",
        "--min-frequency",
        "1",
        "-o",
        str(model),
        "--stats",
    ])
    stats = json.loads(trained.stderr)
    assert stats["unit"] == "grapheme"
    assert stats["pretokenizer"] == "whitespace"
    assert stats["merges"] > 0

    encoded = run_cmd([str(ROOT / "bin" / "dm.exe"), "gpe", "encode", "-m", str(model), "-i", str(corpus)]).stdout.strip().splitlines()
    assert encoded == ["வணக்கம் உலகம்", "வணக்கம் நண்பா"]

    metrics = json.loads(run_cmd([str(ROOT / "bin" / "dm_gpe"), "evaluate", "-m", str(model), "-i", str(corpus)]).stdout)
    assert metrics["compression_ratio"] > 0
    assert metrics["tokenized_length"] == 4

    pre = json.loads(run_cmd([str(ROOT / "bin" / "dm.exe"), "gpe", "pretoken-eval", "-i", str(corpus), "--pretokenizer", "whitespace"]).stdout)
    assert pre["cr_max"] >= 1


if __name__ == "__main__":
    test_native_units_keep_indic_graphemes()
    with tempfile.TemporaryDirectory() as td:
        test_native_gpe_train_encode_evaluate(Path(td))
    print("grapheme_pair_encoding_c tests passed")
