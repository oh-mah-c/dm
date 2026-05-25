#!/usr/bin/env python3
"""
test_trm.py — Python test runner for Tiny Recursion Model (TRM).

Paper: A. Jolicoeur-Martineau,
       "Less is More: Recursive Reasoning with Tiny Networks",
       arXiv:2510.04871v1, Samsung SAIL Montréal, 2025.

This script:
  1. Builds test_trm and trm_train via cmake.
  2. Runs test_trm (13 C++ structural/numerical tests from the paper).
  3. Smoke-tests trm_train CLI for both att and mlp variants.

C++ tests verify:
  - TRM-Att param count ~7M (Table 1)
  - TRM-MLP param count ~5M (Table 1)
  - Logit shape [B, L, V] from deep_recursion
  - Halt signal shape [B, 1]
  - Latent z shape preserved across recursion
  - Deep supervision: T-1 no-grad + 1 with-grad (Section 2.4)
  - Gradient flow: no NaN/Inf
  - AdamW weight update changes parameters (Section 6)
  - EMA weights differ from live after update (Section 4.7)
  - apply_ema / restore_live roundtrip
  - Checkpoint save/load roundtrip
  - predict() returns [B, L] token indices
  - stable_cross_entropy returns finite scalar (Section 6)
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
    print("[test] cmake build test_trm trm_train ...", end=" ", flush=True)
    cmake_build("test_trm", "trm_train")
    assert (BUILD / "test_trm").exists(),  "test_trm binary not found"
    assert (BUILD / "trm_train").exists(), "trm_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    """Run the 13 C++ structural tests derived from the paper."""
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_trm", timeout=600)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(
            f"test_trm exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_trm_train_variants() -> None:
    """Smoke-test trm_train CLI for both att and mlp variants."""
    print("[test] trm_train CLI smoke test ...", end=" ", flush=True)

    binary = BUILD / "trm_train"

    # Unknown variant must exit non-zero
    r = run_binary(binary, "--variant", "xyz_unknown", timeout=30)
    assert r.returncode != 0, "trm_train must reject unknown variant"

    # Valid variants must NOT raise invalid_argument / "Unknown variant"
    for variant in ("att", "mlp"):
        r = run_binary(binary,
                       "--variant", variant,
                       "--epochs",  "1",
                       "--batch",   "2",
                       "--seqlen",  "9",
                       "--vocab",   "10",
                       "--dim",     "32",
                       "--layers",  "1",
                       "--heads",   "2",
                       "--n",       "1",
                       "--T",       "2",
                       "--nsup",    "2",
                       timeout=60)
        combined = r.stdout + r.stderr
        assert "Unknown variant"  not in combined, \
            f"trm_train rejected valid variant '{variant}'\n{combined}"
        assert "invalid_argument" not in combined, \
            f"trm_train threw invalid_argument for '{variant}'\n{combined}"

    print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_trm_train_variants]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} TRM tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
