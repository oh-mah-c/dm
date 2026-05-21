#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.tokenizer_lab import LabBPEModel, learn_bpe_from_chunks, pretokenize, evaluate_model


def test_punct_splits_attribute_access():
    assert pretokenize(".append", "punct") == [".", "append"]


def test_gpt4_caps_digit_runs():
    assert pretokenize("1000", "gpt4") == ["100", "0"]


def test_identity_can_encode_across_whitespace():
    chunks = {"import numpy as np\n": 5, "import numpy": 3}
    merges = learn_bpe_from_chunks(chunks, vocab_size=80, min_frequency=2)
    model = LabBPEModel("identity", merges)
    toks = model.encode_text("import numpy as np\n")
    assert len(toks) <= len("import numpy as np\n")


if __name__ == "__main__":
    test_punct_splits_attribute_access()
    test_gpt4_caps_digit_runs()
    test_identity_can_encode_across_whitespace()
    print("tokenizer_lab tests passed")
