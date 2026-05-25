#!/usr/bin/env python3
"""
test_llama2.py — Python test runner for Llama 2.

Paper: H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat
       Models," arXiv:2307.09288, 2023.
Reference: karpathy/llama2.c  https://github.com/karpathy/llama2.c

This script:
  1. Builds test_llama2 and llama2_train via cmake.
  2. Runs test_llama2 (14 C++ structural tests from the paper).
  3. Smoke-tests llama2_train CLI for the tiny stories110k config.

C++ tests verify:
  - Forward pass output shape [B, T, vocab_size]              (§2)
  - RMSNorm output shape matches input                        (§2.1)
  - RMSNorm unit norm when weight=1                           (§2.1)
  - GQA: n_kv_heads < n_heads works correctly                 (§2.2)
  - RoPE: attention block runs and returns correct shape      (§2.1)
  - SwiGLU FFN output shape                                   (§2.1)
  - Tied weights: output.weight is tok_embeddings.weight      (§2)
  - Causal LM forward with targets sets last_loss             (§2)
  - stories110k param count ≈ 15M                             (config)
  - Gradient flow: no NaN/Inf                                 (§2)
  - AdamW step changes tok_embeddings weights                 (§2)
  - LR schedule: linear warmup, cosine decay peak at warmup   (§2)
  - Checkpoint save/load round-trip
  - KV-cache forward_one ≈ full forward at last position      (§2)
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
    print("[test] cmake build test_llama2 llama2_train ...", end=" ", flush=True)
    cmake_build("test_llama2", "llama2_train")
    assert (BUILD / "test_llama2").exists(),  "test_llama2 binary not found"
    assert (BUILD / "llama2_train").exists(), "llama2_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_llama2", timeout=300)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"test_llama2 exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_cli_smoke() -> None:
    """Smoke-test llama2_train CLI for the tiny stories110k config."""
    for size in ["stories110k"]:
        print(f"[test] llama2_train CLI size={size} smoke test ...", end=" ", flush=True)
        r = run_binary(BUILD / "llama2_train",
                       "--size",    size,
                       "--epochs",  "1",
                       "--batch",   "2",
                       "--data",    "/nonexistent_data_path",
                       timeout=30)
        combined = r.stdout + r.stderr
        assert "terminate called"   not in combined, \
            f"llama2_train size={size} crashed\n{combined}"
        assert "Segmentation fault" not in combined, \
            f"llama2_train size={size} segfaulted\n{combined}"
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
        print(f"All {passed} Llama 2 tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
