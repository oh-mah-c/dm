#!/usr/bin/env python3
"""Build phi-style textbook prompts/corpora through dm.exe.

This is a thin orchestration wrapper around:
  dm textbook prompts
  dm textbook filter
  dm textbook mix
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", default="./bin/dm.exe")
    ap.add_argument("--topics", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--web-samples")
    ap.add_argument("--generated-jsonl")
    ap.add_argument("--web-corpus")
    ap.add_argument("--code-corpus")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = out_dir / "textbook_prompts.jsonl"

    cmd = [
        args.dm,
        "textbook",
        "prompts",
        "--topics",
        args.topics,
        "-o",
        str(prompts),
        "-n",
        str(args.count),
        "--seed",
        str(args.seed),
    ]
    if args.web_samples:
        cmd += ["--web-samples", args.web_samples]
    run(cmd)

    filtered = None
    if args.generated_jsonl:
        filtered = out_dir / "textbook_filtered.txt"
        run([
            args.dm,
            "textbook",
            "filter",
            "-i",
            args.generated_jsonl,
            "-o",
            str(filtered),
            "--min-score",
            "0.55",
        ])

    if filtered and args.web_corpus and args.code_corpus:
        run([
            args.dm,
            "textbook",
            "mix",
            "--synthetic",
            str(filtered),
            "--web",
            args.web_corpus,
            "--code",
            args.code_corpus,
            "-o",
            str(out_dir / "phi_style_mixed_corpus.txt"),
            "--synthetic-ratio",
            "0.40",
            "--web-ratio",
            "0.40",
            "--code-ratio",
            "0.20",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
