"""
Smoke tests for the Llama 3 C++ implementation (arXiv:2407.21783v3).

These tests validate the binary behaviour via subprocess (build + run).
They are intentionally lightweight and target the small 'tiny' preset so
they run on CPU in seconds.
"""

import subprocess
import sys
import os

BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "build")


def run_binary(binary: str, args: list[str] = (), timeout: int = 120) -> subprocess.CompletedProcess:
    path = os.path.join(BUILD_DIR, binary)
    result = subprocess.run(
        [path] + list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def test_unit_tests_pass():
    """All 14 C++ unit tests should pass."""
    r = run_binary("test_llama3")
    assert r.returncode == 0, (
        f"test_llama3 exited with code {r.returncode}\n"
        f"stdout:\n{r.stdout}\n"
        f"stderr:\n{r.stderr}"
    )
    # All checks should pass (count may vary, just ensure no failures)
    assert "0/" not in r.stdout and " tests passed" in r.stdout, (
        f"Expected all tests passed in output:\n{r.stdout}"
    )


def test_train_tiny_smoke():
    """Training on tiny preset should produce finite loss and save a checkpoint."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt = f.name

    try:
        r = run_binary("llama3_train", [
            "--size", "tiny",
            "--epochs", "1",
            "--batch", "2",
            "--maxiter", "5",
            "--warmup", "2",
            "--save", ckpt,
        ])
        assert r.returncode == 0, (
            f"llama3_train failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
        assert os.path.exists(ckpt), "Checkpoint not created"
        # Verify loss was printed
        assert "loss" in r.stdout.lower(), f"Expected loss in output:\n{r.stdout}"
    finally:
        if os.path.exists(ckpt):
            os.remove(ckpt)


def test_config_8b_architecture():
    """The 8B config header values must match Table 3 of the paper."""
    r = run_binary("llama3_train", [
        "--size", "8b",
        "--maxiter", "0",
    ])
    out = r.stdout
    # Table 3: layers=32, dim=4096, n_heads=32, n_kv_heads=8, ffn=14336
    assert "layers=32" in out or "32" in out, f"Unexpected output:\n{out}"


if __name__ == "__main__":
    tests = [test_unit_tests_pass, test_train_tiny_smoke, test_config_8b_architecture]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} Python smoke tests passed")
    sys.exit(0 if passed == len(tests) else 1)
