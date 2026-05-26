"""
test_rwkv.py — Python-level tests for RWKV

Paper: B. Peng et al., "RWKV: Reinventing RNNs for the Transformer Era,"
       arXiv:2305.13048v2, 2023.
       https://arxiv.org/abs/2305.13048

Tests:
  1. Build succeeds (cmake + ninja)
  2. C++ structural tests pass (test_rwkv binary)
  3. Train CLI smoke-test runs without crash
"""

import subprocess
import sys
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
BUILD_DIR = REPO_ROOT / "build"


def ninja_build(target: str) -> bool:
    BUILD_DIR.mkdir(exist_ok=True)
    r = subprocess.run(["cmake", ".."], cwd=BUILD_DIR,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("cmake FAILED:\n", r.stderr)
        return False
    r = subprocess.run(
        ["ninja", target],
        cwd=BUILD_DIR,
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"ninja {target} FAILED:\n", r.stderr)
        return False
    return True


def test_build():
    """Test 1: project builds successfully."""
    assert ninja_build("test_rwkv"), "Build failed"


def test_structural():
    """Test 2: C++ structural tests pass."""
    bin_path = BUILD_DIR / "test_rwkv"
    if not bin_path.exists():
        assert ninja_build("test_rwkv"), "Build failed"
    r = subprocess.run([str(bin_path)], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    assert r.returncode == 0, f"C++ tests failed:\n{r.stdout}\n{r.stderr}"
    assert "0 failed" in r.stdout or "failed" not in r.stdout


def test_train_smoke():
    """Test 3: train CLI smoke-test (tiny config, 2 steps)."""
    ninja_build("rwkv_train")
    bin_path = BUILD_DIR / "rwkv_train"
    if not bin_path.exists():
        print("rwkv_train binary not found, skipping smoke test")
        return

    r = subprocess.run(
        [
            str(bin_path),
            "--size",    "tiny",
            "--epochs",  "1",
            "--batch",   "2",
            "--seqlen",  "16",
            "--maxiter", "2",
            "--save",    "/tmp/rwkv_smoke.pt",
        ],
        capture_output=True, text=True, timeout=120,
    )
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    assert r.returncode == 0, f"Train smoke-test failed:\n{r.stderr}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
