#!/usr/bin/env python3
"""TensorFlow/Numpy Swin reference kernels for fixture generation.

These functions are the golden reference path for the C Swin inference port.
They intentionally use NHWC images and NLC tokens to match the C runtime.
Weights are stored in the C loader layout:

* Dense: [out, in]
* PatchEmbed conv: [out, in, kh, kw]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ensure_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
        return
    except Exception:
        venv_python = ROOT / ".venv" / "bin" / "python"
        if venv_python.exists() and Path(sys.executable) != venv_python:
            os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise


ensure_tensorflow()

import numpy as np
import tensorflow as tf


LAYER_NORM_EPS = 1e-5
MASK_NEGATIVE = -100.0


def as_f32(x):
    return np.asarray(x, dtype=np.float32)


def dense_c(x, weight_out_in, bias=None):
    y = tf.linalg.matmul(tf.convert_to_tensor(x, tf.float32), tf.transpose(tf.convert_to_tensor(weight_out_in, tf.float32)))
    if bias is not None:
        y = y + tf.convert_to_tensor(bias, tf.float32)
    return y


def layer_norm(x, gamma, beta, eps=LAYER_NORM_EPS):
    x = tf.convert_to_tensor(x, tf.float32)
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * tf.math.rsqrt(var + eps) * gamma + beta


def gelu(x):
    return tf.nn.gelu(tf.convert_to_tensor(x, tf.float32), approximate=False)


def softmax_last_dim(x):
    return tf.nn.softmax(tf.convert_to_tensor(x, tf.float32), axis=-1)


def patch_embed_nhwc(x, weight_oihw, bias, patch_size, norm_weight=None, norm_bias=None):
    kernel = tf.transpose(tf.convert_to_tensor(weight_oihw, tf.float32), [2, 3, 1, 0])
    y = tf.nn.conv2d(tf.convert_to_tensor(x, tf.float32), kernel, strides=[1, patch_size, patch_size, 1], padding="VALID")
    y = y + tf.convert_to_tensor(bias, tf.float32)
    b, hp, wp, c = y.shape
    y = tf.reshape(y, [b, hp * wp, c])
    if norm_weight is not None:
        y = layer_norm(y, tf.convert_to_tensor(norm_weight, tf.float32), tf.convert_to_tensor(norm_bias, tf.float32))
    return y


def window_partition(x, window_size):
    x = tf.convert_to_tensor(x, tf.float32)
    b, h, w, c = x.shape
    x = tf.reshape(x, [b, h // window_size, window_size, w // window_size, window_size, c])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, window_size, window_size, c])


def window_reverse(windows, window_size, h, w):
    windows = tf.convert_to_tensor(windows, tf.float32)
    n_windows = int(windows.shape[0])
    b = n_windows // ((h // window_size) * (w // window_size))
    x = tf.reshape(windows, [b, h // window_size, w // window_size, window_size, window_size, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [b, h, w, -1])


def relative_position_index(window_size):
    coords_h = np.arange(window_size, dtype=np.int64)
    coords_w = np.arange(window_size, dtype=np.int64)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = coords.reshape(2, -1)
    rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    rel = np.transpose(rel, [1, 2, 0])
    rel[:, :, 0] += window_size - 1
    rel[:, :, 1] += window_size - 1
    rel[:, :, 0] *= 2 * window_size - 1
    return (rel.sum(-1)).astype(np.int32)


def shifted_attention_mask(h, w, window_size, shift_size):
    img_mask = np.zeros((1, h, w, 1), dtype=np.float32)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).numpy().reshape(-1, window_size * window_size)
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    return np.where(attn_mask != 0, MASK_NEGATIVE, 0.0).astype(np.float32)


def window_attention(x, qkv_weight, qkv_bias, proj_weight, proj_bias, rel_bias_table, num_heads, mask=None):
    x = tf.convert_to_tensor(x, tf.float32)
    b_windows = int(x.shape[0])
    n = int(x.shape[1])
    c = int(x.shape[2])
    head_dim = c // num_heads
    scale = head_dim ** -0.5
    qkv = dense_c(x, qkv_weight, qkv_bias)
    qkv = tf.reshape(qkv, [b_windows, n, 3, num_heads, head_dim])
    qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * scale
    attn = tf.linalg.matmul(q, k, transpose_b=True)
    window_size = int(round(n ** 0.5))
    rel_index = relative_position_index(window_size).reshape(-1)
    rel = tf.gather(tf.convert_to_tensor(rel_bias_table, tf.float32), rel_index)
    rel = tf.reshape(rel, [n, n, num_heads])
    rel = tf.transpose(rel, [2, 0, 1])
    attn = attn + rel[None, :, :, :]
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.float32)
        n_w = int(mask.shape[0])
        attn = tf.reshape(attn, [b_windows // n_w, n_w, num_heads, n, n])
        attn = attn + mask[None, :, None, :, :]
        attn = tf.reshape(attn, [-1, num_heads, n, n])
    attn = tf.nn.softmax(attn, axis=-1)
    y = tf.linalg.matmul(attn, v)
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, [b_windows, n, c])
    return dense_c(y, proj_weight, proj_bias)


def patch_merging(x, h, w, norm_weight, norm_bias, reduction_weight):
    x = tf.convert_to_tensor(x, tf.float32)
    b, l, c = x.shape
    assert int(l) == h * w
    x = tf.reshape(x, [b, h, w, c])
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = tf.concat([x0, x1, x2, x3], axis=-1)
    x = tf.reshape(x, [b, -1, 4 * int(c)])
    x = layer_norm(x, norm_weight, norm_bias)
    return dense_c(x, reduction_weight, None)


def dump_bin(out_dir: Path, records: list[dict], name: str, value, dtype="float32"):
    out_dir.mkdir(parents=True, exist_ok=True)
    if dtype == "int32":
        arr = np.asarray(value, dtype=np.int32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    file = f"{name}.bin"
    arr.tofile(out_dir / file)
    records.append({"name": name, "shape": list(arr.shape), "dtype": dtype, "file": file})
    return arr


def write_metadata(out_dir: Path, records: list[dict], extra: dict | None = None):
    meta = {
        "reference": "TensorFlow/Keras",
        "tensorflow_version": tf.__version__,
        "layout": {
            "image": "NHWC",
            "tokens": "NLC",
            "windows": "[B*nW, M*M, C]",
            "dense_weight": "[out, in]",
            "patch_embed_weight": "[out, in, kh, kw]",
        },
        "layer_norm_epsilon": LAYER_NORM_EPS,
        "gelu": "tf.nn.gelu(approximate=False)",
        "attention_mask_negative": MASK_NEGATIVE,
        "tensors": records,
    }
    if extra:
        meta.update(extra)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
