#!/usr/bin/env python3
"""
test_swin.py — Python test runner for Swin Transformer.

Paper: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang,
       Stephen Lin, Baining Guo,
       "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
       ICCV 2021.  https://arxiv.org/abs/2103.14030

This script:
  1. Builds test_swin and swin_train via cmake.
  2. Runs test_swin (13 C++ structural/numerical tests from the paper).
  3. Smoke-tests swin_train CLI for all 4 model variants.

C++ tests verify:
  - Output shape [N, num_classes] for all 4 configs (Section 3.3)
  - Swin-T/S/B parameter counts match Table 1 (±20%)
  - Patch embedding shape [B, (H/4)*(W/4), C] (Section 3.1)
  - Model has exactly 4 stages (Figure 3)
  - Window partition + reverse roundtrip (Section 3.2)
  - SW-MSA blocks have shift_size = M/2 (Section 3.2)
  - Gradient flow: no NaN/Inf
  - AdamW training step changes weights (Section 4.1)
  - Eval mode determinism
  - Checkpoint save/load round-trip
  - swin_train_epoch / swin_evaluate returns finite values
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
    print("[test] cmake build test_swin swin_train ...", end=" ", flush=True)
    cmake_build("test_swin", "swin_train")
    assert (BUILD / "test_swin").exists(),  "test_swin binary not found"
    assert (BUILD / "swin_train").exists(), "swin_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    """Run the 13 C++ tests derived from the paper."""
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_swin", timeout=600)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(
            f"test_swin exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_swin_train_variants() -> None:
    """Smoke-test swin_train CLI for all 4 variants."""
    print("[test] swin_train CLI smoke test ...", end=" ", flush=True)

    binary = BUILD / "swin_train"

    # Unknown model must exit non-zero
    r = run_binary(binary, "--model", "swin_xyz_unknown", timeout=30)
    assert r.returncode != 0, "swin_train must reject unknown model"

    # Valid models must NOT raise invalid_argument / "Unknown model"
    for variant in ("swin_t", "swin_s", "swin_b", "swin_l"):
        r = run_binary(binary,
                       "--model",  variant,
                       "--epochs", "1",
                       "--batch",  "2",
                       "--data",   "/nonexistent_data_path",
                       timeout=30)
        combined = r.stdout + r.stderr
        assert "Unknown model"    not in combined, \
            f"swin_train rejected valid model '{variant}'\n{combined}"
        assert "invalid_argument" not in combined, \
            f"swin_train threw invalid_argument for '{variant}'\n{combined}"
        # Failure on missing MNIST data is expected and fine

    print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_swin_train_variants]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} Swin Transformer tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
