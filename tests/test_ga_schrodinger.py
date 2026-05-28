#!/usr/bin/env python3
"""Repo-style test runner for GA Schrödinger solver."""

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
    print("[test] cmake build test_ga_schrodinger ga_schrodinger_train ...", end=" ", flush=True)
    cmake_build("test_ga_schrodinger", "ga_schrodinger_train")
    assert (BUILD / "test_ga_schrodinger").exists(), "test_ga_schrodinger binary not found"
    assert (BUILD / "ga_schrodinger_train").exists(), "ga_schrodinger_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    print("[test] C++ GA Schrödinger structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_ga_schrodinger", timeout=180)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"test_ga_schrodinger exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_train_cli_smoke() -> None:
    print("[test] ga_schrodinger_train CLI smoke test ...", end=" ", flush=True)
    r = run_binary(
        BUILD / "ga_schrodinger_train",
        "--system", "box",
        "--grid", "24",
        "--population", "16",
        "--generations", "4",
        "--threshold", "2.0",
        "--seed", "3",
        timeout=120,
    )
    combined = r.stdout + r.stderr
    assert r.returncode == 0, combined
    assert "best_fitness=" in combined, combined
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
        print(f"All {passed} GA Schrödinger tests PASSED")
        sys.exit(0)
    print(f"{passed} passed, {failed} FAILED")
    sys.exit(1)
