"""
Smoke tests for the OhmC2 CPU-first LLM implementation.
"""

import os
import json
import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"


def run(cmd, cwd=REPO_ROOT, timeout=180):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          errors="replace", timeout=timeout)


def build_target(target: str):
    BUILD_DIR.mkdir(exist_ok=True)
    r = run(["cmake", ".."], cwd=BUILD_DIR, timeout=180)
    assert r.returncode == 0, f"cmake failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    r = run(["cmake", "--build", ".", "--target", target, "--parallel", str(os.cpu_count() or 2)],
            cwd=BUILD_DIR, timeout=600)
    assert r.returncode == 0, f"build {target} failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"


def test_unit_binary():
    build_target("test_ohmc2")
    r = run([str(BUILD_DIR / "test_ohmc2")], timeout=180)
    print(r.stdout)
    assert r.returncode == 0, f"test_ohmc2 failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "0 failed" in r.stdout


def test_train_and_infer_smoke():
    build_target("ohmc2_train")
    build_target("ohmc2_infer")
    ckpt = pathlib.Path(tempfile.gettempdir()) / "ohmc2_smoke.pt"
    if ckpt.exists():
        ckpt.unlink()
    meta = pathlib.Path(str(ckpt) + ".json")
    if meta.exists():
        meta.unlink()
    r = run([
        str(BUILD_DIR / "ohmc2_train"),
        "--size", "cpu_quality",
        "--dataset", "datasets/text/input.txt",
        "--seqlen", "8",
        "--max-steps", "2",
        "--eval-interval", "1",
        "--batch", "1",
        "--save", str(ckpt),
    ], timeout=240)
    print(r.stdout)
    assert r.returncode == 0, f"ohmc2_train failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert ckpt.exists(), "checkpoint was not created"
    assert meta.exists(), "checkpoint metadata was not created"
    metadata = json.loads(meta.read_text())
    assert metadata["ffn_type"] == "swiglu"
    assert metadata["use_rope"] is True

    r = run([
        str(BUILD_DIR / "ohmc2_infer"),
        "--model", str(ckpt),
        "--prompt", "Hi",
        "--tokens", "2",
        "--temp", "0",
    ], timeout=120)
    print(r.stdout)
    assert r.returncode == 0, f"ohmc2_infer failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OhmC2 generated text:" in r.stdout

    r = run([
        str(BUILD_DIR / "ohmc2_infer"),
        "--model", str(ckpt),
        "--dim", "64",
        "--prompt", "Hi",
        "--tokens", "1",
    ], timeout=120)
    assert r.returncode != 0, "metadata mismatch should fail"
    assert "metadata mismatch" in r.stderr.lower()


if __name__ == "__main__":
    tests = [test_unit_binary, test_train_and_infer_smoke]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} OhmC2 Python smoke tests passed")
    sys.exit(0 if passed == len(tests) else 1)
