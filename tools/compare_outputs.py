#!/usr/bin/env python3
"""Compare float32 binary outputs and report worst mismatch."""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expected")
    parser.add_argument("actual")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    exp = np.fromfile(args.expected, dtype=np.float32)
    act = np.fromfile(args.actual, dtype=np.float32)
    if exp.shape != act.shape:
        raise SystemExit(f"shape/count mismatch: expected {exp.shape}, actual {act.shape}")
    diff = np.abs(exp - act)
    rel = diff / np.maximum(np.abs(exp), 1e-12)
    idx = int(np.argmax(diff)) if diff.size else 0
    ok = np.allclose(exp, act, atol=args.atol, rtol=args.rtol)
    print(f"max_abs_error={float(diff[idx]) if diff.size else 0.0:.9g}")
    print(f"max_rel_error={float(rel[idx]) if rel.size else 0.0:.9g}")
    print(f"worst_index={idx}")
    if diff.size:
        print(f"expected={float(exp[idx]):.9g}")
        print(f"actual={float(act[idx]):.9g}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
