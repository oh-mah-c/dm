#!/usr/bin/env python3
"""
test_mobilenet.py — Python test runner for MobileNet v1.

Paper: A.G. Howard, M. Zhu, B. Chen et al.,
       "MobileNets: Efficient Convolutional Neural Networks for Mobile
        Vision Applications", arXiv:1704.04861v1, 2017.

This script:
  1. Builds test_mobilenet and mobilenet_train via cmake.
  2. Runs test_mobilenet (14 C++ structural tests from the paper).
  3. Smoke-tests mobilenet_train CLI for each width multiplier.

C++ tests verify:
  - DS block stride=1 preserves spatial dims  (Section 3.1)
  - DS block stride=2 halves spatial dims  (Section 3.1)
  - MobileNet α=1 output [B, 1000]  (Table 1)
  - α=1.0 params ≈ 4.2M  (Table 1)
  - α=0.75 params ≈ 2.6M  (Table 6)
  - α=0.5 params ≈ 1.3M  (Table 6)
  - α=0.25 params ≈ 0.5M  (Table 6)
  - Width multiplier scales params ~quadratically  (Section 3.3)
  - DW conv groups = in_ch  (Section 3.1)
  - DW and PW convs have no bias  (Section 3.2)
  - Gradient flow: no NaN/Inf  (Section 3.2)
  - RMSprop step changes weights  (Section 3.2)
  - Checkpoint save/load round-trip
  - All α variants produce [B, 1000] output
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
    print("[test] cmake build test_mobilenet mobilenet_train ...", end=" ", flush=True)
    cmake_build("test_mobilenet", "mobilenet_train")
    assert (BUILD / "test_mobilenet").exists(),  "test_mobilenet binary not found"
    assert (BUILD / "mobilenet_train").exists(), "mobilenet_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_mobilenet", timeout=300)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"test_mobilenet exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_cli_smoke() -> None:
    """Smoke-test mobilenet_train CLI for α=1.0 and α=0.5."""
    for alpha in ["1.0", "0.5"]:
        print(f"[test] mobilenet_train CLI α={alpha} smoke test ...", end=" ", flush=True)
        r = run_binary(BUILD / "mobilenet_train",
                       "--alpha",   alpha,
                       "--classes", "10",
                       "--epochs",  "1",
                       "--batch",   "2",
                       "--data",    "/nonexistent_data_path",
                       timeout=30)
        combined = r.stdout + r.stderr
        assert "terminate called"   not in combined, \
            f"mobilenet_train α={alpha} crashed\n{combined}"
        assert "Segmentation fault" not in combined, \
            f"mobilenet_train α={alpha} segfaulted\n{combined}"
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
        print(f"All {passed} MobileNet tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
