#!/usr/bin/env python3
"""
test_yolo.py — Python test runner for YOLO v1.

Paper: J. Redmon, S. Divvala, R. Girshick, A. Farhadi,
       "You Only Look Once: Unified, Real-Time Object Detection",
       CVPR 2016.

This script:
  1. Builds test_yolo and yolo_train via cmake.
  2. Runs test_yolo (14 C++ structural tests from the paper).
  3. Smoke-tests yolo_train CLI.

C++ tests verify:
  - Output tensor shape [batch, S, S, B*5+C]  (Section 2)
  - PASCAL VOC: B*5+C = 30 channels  (Section 2)
  - Backbone reduces 448×448 → 7×7  (Figure 3)
  - Leaky ReLU slope = 0.1  (Section 2.2, Eq. 2)
  - Loss is finite and >= 0  (Section 2.2, Eq. 3)
  - λ_coord=5 amplifies coordinate loss  (Section 2.2)
  - λ_noobj=0.5 reduces no-object confidence loss  (Section 2.2)
  - √w/√h trick: large-box error < small-box error  (Section 2.2)
  - Gradient flow: no NaN/Inf
  - SGD+momentum step changes weights  (Section 2.2)
  - box_iou_xywh: identical boxes → IoU = 1
  - NMS removes duplicate boxes  (Section 2.3)
  - decode() returns [N, 6] boxes  (Section 2.3)
  - Checkpoint save/load round-trip
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
    print("[test] cmake build test_yolo yolo_train ...", end=" ", flush=True)
    cmake_build("test_yolo", "yolo_train")
    assert (BUILD / "test_yolo").exists(),  "test_yolo binary not found"
    assert (BUILD / "yolo_train").exists(), "yolo_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    """Run the 14 C++ structural tests derived from the paper."""
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_yolo", timeout=300)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(
            f"test_yolo exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_yolo_train_cli() -> None:
    """Smoke-test yolo_train CLI."""
    print("[test] yolo_train CLI smoke test ...", end=" ", flush=True)
    binary = BUILD / "yolo_train"
    r = run_binary(binary,
                   "--S",      "7",
                   "--B",      "2",
                   "--C",      "20",
                   "--epochs", "1",
                   "--batch",  "2",
                   "--data",   "/nonexistent_data_path",
                   timeout=30)
    combined = r.stdout + r.stderr
    assert "terminate called" not in combined, \
        f"yolo_train crashed unexpectedly\n{combined}"
    assert "Segmentation fault" not in combined, \
        f"yolo_train segfaulted\n{combined}"
    print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_yolo_train_cli]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} YOLO tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
