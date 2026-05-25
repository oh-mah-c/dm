#!/usr/bin/env python3
"""
test_word2vec.py — Python test runner for Word2Vec.

Paper: T. Mikolov, K. Chen, G. Corrado, J. Dean,
       "Efficient Estimation of Word Representations in Vector Space",
       arXiv:1301.3781v3, 2013.

This script:
  1. Builds test_word2vec and word2vec_train via cmake.
  2. Runs test_word2vec (14 C++ structural tests from the paper).
  3. Smoke-tests word2vec_train CLI (skip-gram and CBOW).

C++ tests verify:
  - W_in / W_out shapes [V, D]
  - CBOW forward output shape [batch, V]  (Section 3.1)
  - Skip-gram forward output shape [batch]  (Section 3.2)
  - W_in init U[-0.5/D, 0.5/D]; W_out init zeros
  - NS loss is finite and >= 0
  - NS loss decreases for well-separated scores
  - CBOW context embeddings averaged  (Section 3.1)
  - cosine_similarity: same word → 1.0
  - analogy: vec(a)-vec(b)+vec(c) ≈ vec(d)  (Section 4)
  - most_similar: k results, excludes query word
  - SGD step changes W_in  (Section 4)
  - build_pairs SkipGram: [N] centre + [N] context  (Section 3.2)
  - build_pairs CBOW: [N, 2w] context + [N] centre  (Section 3.1)
  - Checkpoint save/load round-trip
"""

import subprocess
import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def cmake_build(*targets: str) -> None:
    cmd = ["cmake", "--build", str(BUILD), "--parallel", "4",
           "--target", *targets]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"cmake build failed for: {targets}")


def run_binary(binary: Path, *args: str,
               timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *args],
                          cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build() -> None:
    print("[test] cmake build test_word2vec word2vec_train ...", end=" ", flush=True)
    cmake_build("test_word2vec", "word2vec_train")
    assert (BUILD / "test_word2vec").exists(),   "test_word2vec binary not found"
    assert (BUILD / "word2vec_train").exists(),  "word2vec_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_word2vec", timeout=120)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"test_word2vec exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_cli_skipgram() -> None:
    print("[test] word2vec_train CLI skip-gram smoke test ...", end=" ", flush=True)
    binary = BUILD / "word2vec_train"
    r = run_binary(binary,
                   "--mode",   "skipgram",
                   "--dim",    "32",
                   "--window", "2",
                   "--neg",    "3",
                   "--epochs", "1",
                   "--batch",  "64",
                   "--data",   "/nonexistent_data_path",
                   timeout=30)
    combined = r.stdout + r.stderr
    assert "terminate called"  not in combined, f"skip-gram crashed\n{combined}"
    assert "Segmentation fault" not in combined, f"skip-gram segfaulted\n{combined}"
    print("PASS")


def test_cli_cbow() -> None:
    print("[test] word2vec_train CLI CBOW smoke test ...", end=" ", flush=True)
    binary = BUILD / "word2vec_train"
    r = run_binary(binary,
                   "--mode",   "cbow",
                   "--dim",    "32",
                   "--window", "2",
                   "--neg",    "3",
                   "--epochs", "1",
                   "--batch",  "64",
                   "--data",   "/nonexistent_data_path",
                   timeout=30)
    combined = r.stdout + r.stderr
    assert "terminate called"  not in combined, f"cbow crashed\n{combined}"
    assert "Segmentation fault" not in combined, f"cbow segfaulted\n{combined}"
    print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_cli_skipgram, test_cli_cbow]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} Word2Vec tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
