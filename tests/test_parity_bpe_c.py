#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / "bin" / "dm.exe"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(DM), *args], cwd=ROOT, text=True, capture_output=True, check=True)


def test_train_focus_encode_evaluate_trace() -> None:
    subprocess.run(["make", "bin/dm.exe", "bin/dm_parity_bpe"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        train = t / "train.tsv"
        train.write_text("hi\tabababab\nlo\txyzxyz\n", encoding="utf-8")
        dev = t / "dev.tsv"
        dev.write_text("hi\tabababab\nlo\txyzxyz\n", encoding="utf-8")
        model = t / "pbpe.json"
        run(
            "parity_bpe",
            "train",
            "--input-labeled",
            str(train),
            "--dev-labeled",
            str(dev),
            "--merges",
            "4",
            "--min-frequency",
            "1",
            "--strategy",
            "parity",
            "--keep-trace",
            "-o",
            str(model),
        )
        data = json.loads(model.read_text(encoding="utf-8"))
        assert data["version"] == "dm-parity-aware-bpe-2508.04796v2"
        assert len(data["merges"]) == 4
        assert data["trace"][0]["focus_language"] in {"hi", "lo"}

        text = t / "in.txt"
        text.write_text("abab xyz\n", encoding="utf-8")
        encoded = run("parity_bpe", "encode", "-m", str(model), "-i", str(text), "--hex-tokens").stdout
        assert encoded.startswith("[")
        decoded = subprocess.run(
            [str(DM), "parity_bpe", "decode", "-m", str(model)],
            input=encoded,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        assert decoded == "abab xyz\n"

        metrics = json.loads(run("parity_bpe", "evaluate", "-m", str(model), "--input-labeled", str(train)).stdout)
        assert metrics["tokens"] > 0
        assert "tokenizer_fairness_gini" in metrics
        trace = run("parity_bpe", "trace", "-m", str(model)).stdout
        assert trace.splitlines()[0] == "step,focus_language,pair_count,min_cr_before,gini_before"


if __name__ == "__main__":
    test_train_focus_encode_evaluate_trace()
    print("parity_bpe C tests passed")
