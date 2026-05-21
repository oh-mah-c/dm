#!/usr/bin/env python3
"""Command-line entrypoint for dm's Grapheme Pair Encoding tokenizer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizer.nlp.grapheme_pair_encoding import main


if __name__ == "__main__":
    raise SystemExit(main())
