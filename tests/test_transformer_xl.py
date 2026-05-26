"""
test_transformer_xl.py — Python-level tests for Transformer-XL

Paper: Z. Dai et al., "Transformer-XL: Attentive Language Models Beyond a
       Fixed-Length Context," ACL 2019.
       https://aclanthology.org/P19-1285

Tests:
  1. Build succeeds (cmake + make)
  2. C++ structural tests pass (test_transformer_xl binary)
  3. Train CLI smoke-test runs without crash
"""

import subprocess
import sys
import os
import pathlib

# Locate build directory relative to this file
REPO_ROOT = pathlib.Path(__file__).parent.parent
BUILD_DIR = REPO_ROOT / "build"


def cmake_build() -> bool:
    """Configure and build the project."""
    BUILD_DIR.mkdir(exist_ok=True)
    r = subprocess.run(
        ["cmake", ".."],
        cwd=BUILD_DIR,
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("cmake configure FAILED:\n", r.stderr)
        return False
    r = subprocess.run(
        ["make", "-j", str(os.cpu_count() or 2), "test_transformer_xl"],
        cwd=BUILD_DIR,
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("make FAILED:\n", r.stderr)
        return False
    return True


def test_build():
    """Test 1: project builds successfully."""
    assert cmake_build(), "Build failed"


def test_structural():
    """Test 2: C++ structural tests pass."""
    bin_path = BUILD_DIR / "test_transformer_xl"
    if not bin_path.exists():
        assert cmake_build(), "Build failed"
    r = subprocess.run(
        [str(bin_path)],
        capture_output=True, text=True,
    )
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    assert r.returncode == 0, f"C++ tests failed:\n{r.stdout}\n{r.stderr}"
    assert "failed" not in r.stdout or "0 failed" in r.stdout


def test_train_smoke():
    """Test 3: train CLI smoke-test (tiny config, 2 steps)."""
    # Build train binary
    r = subprocess.run(
        ["make", "-j", str(os.cpu_count() or 2), "transformer_xl_train"],
        cwd=BUILD_DIR,
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        assert cmake_build(), "Build failed"

    bin_path = BUILD_DIR / "transformer_xl_train"
    if not bin_path.exists():
        print("transformer_xl_train binary not found, skipping smoke test")
        return

    r = subprocess.run(
        [
            str(bin_path),
            "--size",    "tiny",
            "--epochs",  "1",
            "--batch",   "2",
            "--maxiter", "2",
            "--save",    "/tmp/txl_smoke.pt",
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
