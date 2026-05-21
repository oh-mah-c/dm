#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / "bin" / "dm.exe"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(DM), *args], cwd=ROOT, text=True, capture_output=True, check=True)


def test_tokenizer_lab_native_flow() -> None:
    subprocess.run(["make", "bin/dm.exe", "bin/dm_tokenizer_lab"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        corpus = t / "code.txt"
        corpus.write_text("x.append(1000)\nimport numpy as np\nx.append(1000)\n", encoding="utf-8")
        model = t / "tok.json"
        run("tokenizer_lab", "train-bpe", "-i", str(corpus), "-o", str(model), "--vocab-size", "80", "--pretokenizer", "punct", "--min-frequency", "1")
        payload = json.loads(model.read_text(encoding="utf-8"))
        assert payload["version"] == "dm-tokenizer-lab-dagan-2024"
        assert payload["pretokenizer"] == "punct"

        encoded = run("tokenizer_lab", "encode", "-m", str(model), "-i", str(corpus), "--json-tokens").stdout
        assert encoded.splitlines()[0].startswith("[")

        metrics = json.loads(run("tokenizer_lab", "evaluate", "-m", str(model), "-i", str(corpus)).stdout)
        assert metrics["tokens"] > 0
        assert "renyi_alpha_2p5" in metrics

        comp = run("tokenizer_lab", "compare", "-m", str(model), "-i", str(corpus)).stdout
        assert comp.splitlines()[0].startswith("model,pretokenizer,tokens")

        csv = t / "nsl.csv"
        csv.write_text("vocab_size,nsl32k\n32000,1.0\n64000,0.8\n", encoding="utf-8")
        trade = json.loads(run("tokenizer_lab", "vocab-tradeoff", "--nsl-csv", str(csv), "--dim", "4096", "--layers", "32", "--heads", "32", "--kv-heads", "8").stdout)
        assert "memory_optimal_vocab_size" in trade


if __name__ == "__main__":
    test_tokenizer_lab_native_flow()
    print("tokenizer_lab C tests passed")
