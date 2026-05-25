#!/usr/bin/env python3
"""
test_whisper.py — Python test runner for Whisper.

Paper: A. Radford et al.,
       "Robust Speech Recognition via Large-Scale Weak Supervision"
       ICML 2023 (radford23a)

This script:
  1. Builds test_whisper and whisper_train via cmake.
  2. Runs test_whisper (14 C++ structural tests from the paper).
  3. Smoke-tests whisper_train CLI for tiny and base sizes.

C++ tests verify:
  - Encoder output shape [B, T/2, d_model]                (Section 2.1)
  - Encoder halves time dimension (conv stride=2)          (Section 2.1)
  - Decoder logits shape [B, L, vocab_size]                (Section 2.1)
  - Full forward pass shape [B, L, vocab_size]             (Section 2.1)
  - Tiny model params ≈ 39M                                (Table 1)
  - Base model params ≈ 74M                                (Table 1)
  - Encoder block count matches enc_layers                 (Section 2.1)
  - Decoder block count matches dec_layers                 (Section 2.1)
  - attention head_dim = d_model / n_heads                 (Section 2.1)
  - Positional embedding shape [1, max_src_pos, d_model]   (Section 2.1)
  - Gradient flow: no NaN/Inf                              (Section 2.2)
  - AdamW step changes encoder weights                     (Section 2.2)
  - Checkpoint save/load round-trip
  - All factory variants produce correct output shape
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
               timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *args],
                          cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build() -> None:
    print("[test] cmake build test_whisper whisper_train ...", end=" ", flush=True)
    cmake_build("test_whisper", "whisper_train")
    assert (BUILD / "test_whisper").exists(),  "test_whisper binary not found"
    assert (BUILD / "whisper_train").exists(), "whisper_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_whisper", timeout=300)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"test_whisper exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_cli_smoke() -> None:
    """Smoke-test whisper_train CLI for tiny and base sizes."""
    for size in ["tiny", "base"]:
        print(f"[test] whisper_train CLI size={size} smoke test ...", end=" ", flush=True)
        r = run_binary(BUILD / "whisper_train",
                       "--size",   size,
                       "--epochs", "1",
                       "--batch",  "2",
                       "--data",   "/nonexistent_data_path",
                       timeout=30)
        combined = r.stdout + r.stderr
        assert "terminate called"   not in combined, \
            f"whisper_train size={size} crashed\n{combined}"
        assert "Segmentation fault" not in combined, \
            f"whisper_train size={size} segfaulted\n{combined}"
        print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_cli_smoke]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} Whisper tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
