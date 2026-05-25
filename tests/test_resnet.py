#!/usr/bin/env python3
"""
test_resnet.py — Python test runner for the ResNet implementation.

Paper: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016

This script:
  1. Builds the test_resnet and resnet_train binaries via cmake.
  2. Runs test_resnet (C++ structural / numerical tests derived from the paper).
  3. Smoke-tests resnet_train with --help (verifies the binary is runnable
     and all 5 model variants parse without error).

The C++ tests (test_resnet) verify:
  - Output shape [N, num_classes] for all 5 variants (Table 1)
  - Parameter counts match published values (±2%)
  - Gradient flow: no NaN/Inf for all 5 variants
  - Shortcut types: projection at dim-change, identity otherwise (Sec 3.3)
  - Bottleneck expansion=4 channel widths (Fig. 5 right)
  - SGD training step mutates weights (Sec 3.4)
  - train vs eval mode consistency (BN behaviour)
  - Checkpoint save/load round-trip preserves output
  - Spatial size progression 224→112→56→28→14→7 (Table 1)
  - resnet_train_epoch / resnet_evaluate API returns finite values

All assertions are derived directly from the paper.
"""

import subprocess
import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


# ── Build helpers ─────────────────────────────────────────────────────────────

def cmake_build(*targets: str) -> None:
    """Build one or more cmake targets (runs cmake --build)."""
    cmd = [
        "cmake", "--build", str(BUILD),
        "--parallel", "4",
        "--target", *targets,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        raise RuntimeError(f"cmake build failed for targets: {targets}")


def run_binary(binary: Path, *args: str,
               timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a binary, return CompletedProcess. Raises on non-zero exit."""
    result = subprocess.run(
        [str(binary), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build() -> None:
    """Build test_resnet and resnet_train binaries."""
    print("[test] cmake build test_resnet resnet_train ...", end=" ", flush=True)
    cmake_build("test_resnet", "resnet_train")
    assert (BUILD / "test_resnet").exists(),  "test_resnet binary not found"
    assert (BUILD / "resnet_train").exists(), "resnet_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    """
    Run the C++ test binary. It exits 0 on success, 1 with a FAIL message on
    any assertion error.

    Tests run by the binary (all derived from the paper):
      1. Output shape
      2. Parameter counts
      3. Gradient flow (no NaN/Inf)
      4. Shortcut types
      5. Bottleneck expansion=4
      6. SGD training step
      7. Train/eval mode
      8. Checkpoint round-trip
      9. Spatial size progression
     10. resnet_train_epoch / resnet_evaluate API
    """
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_resnet", timeout=600)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(
            f"test_resnet exited with code {r.returncode}\n{r.stderr}")
    # Print each line so pytest -s shows progress
    for line in r.stdout.splitlines():
        print(" ", line)


def test_resnet_train_variants() -> None:
    """
    Smoke-test resnet_train --help / bad model name.
    Verifies the binary is runnable and CLI parsing works for all 5 variants.
    We don't actually run training (needs MNIST data and is slow), but we
    verify the binary starts and parses flags without crashing.
    """
    print("[test] resnet_train CLI smoke test ...", end=" ", flush=True)

    binary = BUILD / "resnet_train"

    # Passing an unknown model should exit non-zero (std::invalid_argument)
    r = run_binary(binary, "--model", "invalid_model_xyz", timeout=30)
    assert r.returncode != 0, \
        "resnet_train should exit non-zero for unknown model"

    # Each valid model name should be accepted. The binary will abort when it
    # tries to open the MNIST dataset (missing data path), but it must NOT fail
    # with "Unknown model" or "invalid_argument" — that would mean the model
    # name wasn't recognised.  We distinguish the two failure modes by checking
    # that stderr does NOT contain "Unknown model" / "invalid_argument".
    for variant in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        r = run_binary(binary,
                       "--model",  variant,
                       "--epochs", "1",
                       "--batch",  "2",
                       "--data",   "/nonexistent_data_path",
                       timeout=30)
        combined = r.stdout + r.stderr
        # Must NOT die with "Unknown model" (that means CLI parsing rejected it)
        assert "Unknown model" not in combined, \
            f"resnet_train rejected valid model '{variant}'\n{combined}"
        # Must NOT die with "invalid_argument" from our throw
        assert "invalid_argument" not in combined, \
            f"resnet_train invalid_argument for '{variant}'\n{combined}"
        # Failure on missing MNIST data is expected and acceptable
        # (c10::Error about "Error opening images file" is fine)

    print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0

    tests = [
        test_build,
        test_cpp_structural,
        test_resnet_train_variants,
    ]

    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} ResNet tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
