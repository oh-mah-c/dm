#!/usr/bin/env python3
"""
test_vgg.py — Python test runner for VGGNet.

Paper: Simonyan & Zisserman,
       "Very Deep Convolutional Networks for Large-Scale Image Recognition",
       ICLR 2015  (arXiv:1409.1556v6)

This script:
  1. Builds test_vgg and vgg_train via cmake.
  2. Runs test_vgg (13 C++ structural/numerical tests from the paper).
  3. Smoke-tests vgg_train CLI for all 5 model variants.

C++ tests verify:
  - Output shape [N, num_classes] for all 5 configs (Table 1)
  - Weight layer counts  A=11, B=13, C=16, D=16, E=19 (Table 1)
  - Parameter counts match Table 2: ~133M / 133M / 134M / 138M / 144M (±5%)
  - Feature block structure: 5 maxpool layers per config
  - Config C has exactly 3 conv1×1 layers; D and E have none
  - Spatial size progression 224→112→56→28→14→7 (Section 2.1)
  - No LRN layers (Section 2.1: "none of our networks uses LRN")
  - Exactly 2 Dropout(0.5) layers in classifier (Section 3.1)
  - Gradient flow: no NaN/Inf for vgg_a, vgg16, vgg19
  - SGD training step changes weights (Section 3.1)
  - Eval mode determinism
  - Checkpoint save/load round-trip
  - vgg_train_epoch / vgg_evaluate returns finite values
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
    print("[test] cmake build test_vgg vgg_train ...", end=" ", flush=True)
    cmake_build("test_vgg", "vgg_train")
    assert (BUILD / "test_vgg").exists(),  "test_vgg binary not found"
    assert (BUILD / "vgg_train").exists(), "vgg_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    """Run the 13 C++ tests derived from the paper."""
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_vgg", timeout=600)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(
            f"test_vgg exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_vgg_train_variants() -> None:
    """Smoke-test vgg_train CLI for all 5 variants."""
    print("[test] vgg_train CLI smoke test ...", end=" ", flush=True)

    binary = BUILD / "vgg_train"

    # Unknown model must exit non-zero
    r = run_binary(binary, "--model", "vgg_xyz_unknown", timeout=30)
    assert r.returncode != 0, "vgg_train must reject unknown model"

    # Valid models must NOT raise invalid_argument / "Unknown model"
    for variant in ("vgg_a", "vgg_b", "vgg_c", "vgg16", "vgg19"):
        r = run_binary(binary,
                       "--model",  variant,
                       "--epochs", "1",
                       "--batch",  "2",
                       "--data",   "/nonexistent_data_path",
                       timeout=30)
        combined = r.stdout + r.stderr
        assert "Unknown model"    not in combined, \
            f"vgg_train rejected valid model '{variant}'\n{combined}"
        assert "invalid_argument" not in combined, \
            f"vgg_train threw invalid_argument for '{variant}'\n{combined}"
        # Failure on missing MNIST data is expected and fine

    print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_vgg_train_variants]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} VGGNet tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
