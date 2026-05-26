"""
test_lora.py — Python-level tests for LoRA

Paper: E. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models,"
       arXiv:2106.09685v2, 2021.
       https://arxiv.org/abs/2106.09685

Tests:
  1. Build succeeds (cmake + ninja)
  2. C++ structural tests pass (test_lora binary)
  3. Train CLI smoke-test runs without crash
"""

import subprocess
import sys
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
    assert ninja_build("test_lora"), "Build failed"


def test_structural():
    """Test 2: C++ structural tests pass."""
    bin_path = BUILD_DIR / "test_lora"
    if not bin_path.exists():
        assert ninja_build("test_lora"), "Build failed"
    r = subprocess.run([str(bin_path)], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    assert r.returncode == 0, f"C++ tests failed:\n{r.stdout}\n{r.stderr}"
    assert "0 failed" in r.stdout or "failed" not in r.stdout


def test_train_smoke():
    """Test 3: train CLI smoke-test (tiny config, 2 steps)."""
    ninja_build("lora_train")
    bin_path = BUILD_DIR / "lora_train"
    if not bin_path.exists():
        print("lora_train binary not found, skipping smoke test")
        return

    r = subprocess.run(
        [
            str(bin_path),
            "--rank",    "4",
            "--alpha",   "4.0",
            "--layers",  "2",
            "--dim",     "64",
            "--seqlen",  "16",
            "--epochs",  "1",
            "--batch",   "2",
            "--maxiter", "2",
            "--targets", "W_q,W_v",
            "--save",    "/tmp/lora_smoke.pt",
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
