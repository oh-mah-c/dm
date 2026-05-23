#!/usr/bin/env python3
"""
dm_resnet.py — Python bindings for the ResNet-18 implementation in libdm.

Usage:
    python examples/python/dm_resnet.py [--size 224] [--classes 1000] [--seed 42]

Requires:
    DM_LIB=/path/to/libdm.so (or place libdm.so in the project root)
    LD_LIBRARY_PATH=/path/to/tensorflow
"""

import ctypes
import os
import sys
import struct
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# C struct mirroring DM_Tensor (NCHW layout, float32 data)
# ─────────────────────────────────────────────────────────────────────────────
class DM_Tensor(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("c", ctypes.c_int),
        ("h", ctypes.c_int),
        ("w", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_float)),
    ]


def _load_lib() -> ctypes.CDLL:
    path = os.environ.get("DM_LIB") or str(ROOT / "libdm.so")
    lib = ctypes.CDLL(path)

    # ── Tensor lifecycle ──────────────────────────────────────────────────────
    lib.dm_tensor_alloc.argtypes = [
        ctypes.POINTER(DM_Tensor), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.dm_tensor_alloc.restype = ctypes.c_int

    lib.dm_tensor_free.argtypes = [ctypes.POINTER(DM_Tensor)]
    lib.dm_tensor_free.restype = None

    lib.dm_tensor_fill.argtypes = [ctypes.POINTER(DM_Tensor), ctypes.c_float]
    lib.dm_tensor_fill.restype = None

    lib.dm_softmax.argtypes = [ctypes.POINTER(DM_Tensor)]
    lib.dm_softmax.restype = None

    # ── Primitive operators ───────────────────────────────────────────────────
    lib.dm_op_relu.argtypes = [ctypes.POINTER(DM_Tensor)]
    lib.dm_op_relu.restype = None

    lib.dm_op_tensor_add.argtypes = [
        ctypes.POINTER(DM_Tensor), ctypes.POINTER(DM_Tensor),
    ]
    lib.dm_op_tensor_add.restype = ctypes.c_int

    lib.dm_op_max_pool2d_same.argtypes = [
        ctypes.POINTER(DM_Tensor), ctypes.POINTER(DM_Tensor),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.dm_op_max_pool2d_same.restype = ctypes.c_int

    lib.dm_op_batch_norm.argtypes = [
        ctypes.POINTER(DM_Tensor),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    lib.dm_op_batch_norm.restype = ctypes.c_int

    # ── ResNet models ─────────────────────────────────────────────────────────
    lib.dm_op_resnet18_forward.argtypes = [
        ctypes.POINTER(DM_Tensor), ctypes.POINTER(DM_Tensor),
        ctypes.c_int, ctypes.c_uint,
    ]
    lib.dm_op_resnet18_forward.restype = ctypes.c_int

    lib.dm_op_resnet_basic_block.argtypes = [
        ctypes.POINTER(DM_Tensor), ctypes.POINTER(DM_Tensor),
        ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]
    lib.dm_op_resnet_basic_block.restype = ctypes.c_int

    return lib


_lib = _load_lib()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tensor_from_numpy(arr: np.ndarray) -> DM_Tensor:
    """Copy a NCHW float32 numpy array into a DM_Tensor."""
    assert arr.ndim == 4, "Expected 4-D NCHW array"
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, c, h, w = arr.shape
    t = DM_Tensor()
    assert _lib.dm_tensor_alloc(ctypes.byref(t), n, c, h, w) == 0, "alloc failed"
    size = n * c * h * w
    src = arr.flatten()
    for i in range(size):
        t.data[i] = float(src[i])
    return t


def _tensor_to_numpy(t: DM_Tensor) -> np.ndarray:
    """Read a DM_Tensor back into a NCHW float32 numpy array."""
    size = t.n * t.c * t.h * t.w
    buf = ctypes.cast(t.data, ctypes.POINTER(ctypes.c_float * size))
    return np.frombuffer(buf.contents, dtype=np.float32).reshape(t.n, t.c, t.h, t.w).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Public Python API
# ─────────────────────────────────────────────────────────────────────────────
def resnet18_forward(
    x: np.ndarray,
    classes: int = 1000,
    seed: int = 42,
    top_k: int = 5,
) -> dict:
    """
    Run a ResNet-18 forward pass.

    Parameters
    ----------
    x : np.ndarray  shape (N, C, H, W), dtype float32
        Input image tensor in NCHW layout, values in [0, 1].
    classes : int
        Number of output classes.
    seed : int
        Deterministic weight seed (same seed → same weights).
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    dict with keys:
        "logits"      : np.ndarray  (N, classes)  raw pre-softmax scores
        "probs"       : np.ndarray  (N, classes)  softmax probabilities
        "top_classes" : list[int]   top-k class indices (highest prob first)
        "top_probs"   : list[float] corresponding probabilities
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.ndim == 4, "Input must be 4-D NCHW"
    n, c, h, w = x.shape

    t_in = _tensor_from_numpy(x)
    t_logits = DM_Tensor()
    assert _lib.dm_tensor_alloc(ctypes.byref(t_logits), n, classes, 1, 1) == 0

    rc = _lib.dm_op_resnet18_forward(
        ctypes.byref(t_in), ctypes.byref(t_logits),
        ctypes.c_int(classes), ctypes.c_uint(seed),
    )
    _lib.dm_tensor_free(ctypes.byref(t_in))
    assert rc == 0, f"dm_op_resnet18_forward returned {rc}"

    logits = _tensor_to_numpy(t_logits).reshape(n, classes)

    # Softmax in Python for clarity
    def _softmax(z):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    probs = _softmax(logits)
    top_idx = np.argsort(-probs[0])[:top_k]

    _lib.dm_tensor_free(ctypes.byref(t_logits))

    return {
        "logits":      logits,
        "probs":       probs,
        "top_classes": top_idx.tolist(),
        "top_probs":   probs[0][top_idx].tolist(),
    }


def resnet_basic_block(
    x: np.ndarray,
    out_channels: int,
    stride: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Run a single ResNet BasicBlock (two 3×3 convs + residual shortcut).

    Parameters
    ----------
    x : np.ndarray  shape (N, C, H, W), dtype float32
    out_channels : int
    stride : int
    seed : int

    Returns
    -------
    np.ndarray  shape (N, out_channels, H//stride, W//stride)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    t_in = _tensor_from_numpy(x)
    t_out = DM_Tensor()

    rc = _lib.dm_op_resnet_basic_block(
        ctypes.byref(t_in), ctypes.byref(t_out),
        ctypes.c_int(out_channels), ctypes.c_int(stride), ctypes.c_uint(seed),
    )
    _lib.dm_tensor_free(ctypes.byref(t_in))
    assert rc == 0, f"dm_op_resnet_basic_block returned {rc}"

    result = _tensor_to_numpy(t_out)
    _lib.dm_tensor_free(ctypes.byref(t_out))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="ResNet-18 Python demo via libdm FFI")
    p.add_argument("--size",    type=int, default=64,   help="Input spatial size (H=W)")
    p.add_argument("--classes", type=int, default=10,   help="Number of output classes")
    p.add_argument("--seed",    type=int, default=42,   help="Weight seed")
    p.add_argument("--topk",   type=int, default=5,    help="Top-k predictions to show")
    args = p.parse_args()

    print(f"ResNet-18 demo  size={args.size}×{args.size}  classes={args.classes}  seed={args.seed}")

    # Synthetic input: ones
    x = np.ones((1, 3, args.size, args.size), dtype=np.float32)

    out = resnet18_forward(x, classes=args.classes, seed=args.seed, top_k=args.topk)

    print(f"\nRaw logit range: [{out['logits'].min():.4f}, {out['logits'].max():.4f}]")
    print(f"\nTop-{args.topk} predictions:")
    for rank, (cls, prob) in enumerate(zip(out["top_classes"], out["top_probs"]), 1):
        print(f"  rank {rank:2d}  class {cls:4d}  prob {prob:.6f}")
