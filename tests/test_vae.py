#!/usr/bin/env python3
"""
test_vae.py — Python test runner for Variational Auto-Encoder.

Paper: D.P. Kingma and M. Welling,
       "Auto-Encoding Variational Bayes",
       arXiv:1312.6114v11, 2022.

This script:
  1. Builds test_vae and vae_train via cmake.
  2. Runs test_vae (13 C++ structural tests from the paper).
  3. Smoke-tests vae_train CLI.

C++ tests verify:
  - Encoder output shapes [B, latent_dim] for μ and log_var (Appendix C.2)
  - Decoder output shape [B, input_dim] in (0,1) (Appendix C.1)
  - Reparameterisation z shape [B, latent_dim] (Section 2.4)
  - Eval mode: z == μ (deterministic, no noise)
  - Forward pass output shapes {recon_x, μ, log_var}
  - ELBO loss is finite and positive (Eq. 10)
  - KL term >= 0 (Appendix B analytical solution)
  - Reconstruction BCE >= 0 (Appendix C.1)
  - Gradient flow: no NaN/Inf
  - Adam optimiser step changes weights (Section 5)
  - sample() returns [N, input_dim] in (0,1)
  - encode() returns [B, latent_dim]
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
    print("[test] cmake build test_vae vae_train ...", end=" ", flush=True)
    cmake_build("test_vae", "vae_train")
    assert (BUILD / "test_vae").exists(),  "test_vae binary not found"
    assert (BUILD / "vae_train").exists(), "vae_train binary not found"
    print("PASS")


def test_cpp_structural() -> None:
    """Run the 13 C++ structural tests derived from the paper."""
    print("[test] C++ structural/numerical tests ...", flush=True)
    r = run_binary(BUILD / "test_vae", timeout=120)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(
            f"test_vae exited {r.returncode}\n{r.stderr}")
    for line in r.stdout.splitlines():
        print(" ", line)


def test_vae_train_cli() -> None:
    """Smoke-test vae_train CLI."""
    print("[test] vae_train CLI smoke test ...", end=" ", flush=True)

    binary = BUILD / "vae_train"
    r = run_binary(binary,
                   "--input-dim",  "64",
                   "--hidden-dim", "32",
                   "--latent-dim", "8",
                   "--epochs",     "1",
                   "--batch",      "4",
                   "--data",       "/nonexistent_data_path",
                   timeout=30)
    combined = r.stdout + r.stderr
    # MNIST missing is fine; model construction must succeed
    assert "terminate called" not in combined, \
        f"vae_train crashed unexpectedly\n{combined}"
    assert "Segmentation fault" not in combined, \
        f"vae_train segfaulted\n{combined}"
    print("PASS")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in [test_build, test_cpp_structural, test_vae_train_cli]:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {passed} VAE tests PASSED")
        sys.exit(0)
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
