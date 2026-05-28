#!/usr/bin/env python3
"""Repo-style test runner for Neural Quantum States."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def cmake_build(*targets: str) -> None:
    cmd = ["cmake", "--build", str(BUILD), "--parallel", "4", "--target", *targets]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"cmake build failed for: {targets}")


def run_binary(binary: Path, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


def test_build() -> None:
    print("[test] cmake build test_nqs nqs_train ...", end=" ", flush=True)
    cmake_build("test_nqs", "nqs_train")
    assert (BUILD / "test_nqs").exists(), "test_nqs binary not found"
    assert (BUILD / "nqs_train").exists(), "nqs_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    print("[test] C++ NQS structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_nqs", timeout=180)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"test_nqs exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_train_cli_smoke() -> None:
    print("[test] nqs_train CLI smoke test ...", end=" ", flush=True)
    r = run_binary(
        BUILD / "nqs_train",
        "--spins", "4",
        "--hidden", "4",
        "--steps", "2",
        "--lr", "0.01",
        "--J", "1.0",
        "--h", "0.5",
        "--save", "/tmp/nqs_smoke.pt",
        timeout=120,
    )
    combined = r.stdout + r.stderr
    assert r.returncode == 0, combined
    assert "nan" not in combined.lower(), combined
    assert "inf" not in combined.lower(), combined
    print("PASS")


if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_train_cli_smoke]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    if failed == 0:
        print(f"All {passed} NQS tests PASSED")
        sys.exit(0)
    print(f"{passed} passed, {failed} FAILED")
    sys.exit(1)
