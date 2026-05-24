#!/usr/bin/env python3
"""Export TensorFlow/Keras Swin-style weights in the C loader layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

def dump_tensor(out_dir: Path, records: list[dict], name: str, arr):
    arr = np.asarray(arr, dtype=np.float32)
    rel = Path("tensors") / f"{name}.bin"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    records.append({"name": name, "shape": list(arr.shape), "dtype": "float32", "file": str(rel).replace("\\", "/")})


def export_synthetic_swin_tiny(out_dir: Path, seed: int):
    """Create deterministic TensorFlow-layout-compatible synthetic Swin-T weights.

    This is not a trained checkpoint. It exists so C/TensorFlow parity tests can
    run without PyTorch or an external checkpoint.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_dim = 96
    depths = [2, 2, 6, 2]
    heads = [3, 6, 12, 24]
    window = 7
    num_classes = 1000

    def rand(shape, scale=0.02):
        return (rng.normal(size=shape) * scale).astype(np.float32)

    def ones(shape):
        return np.ones(shape, dtype=np.float32)

    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    dump_tensor(out_dir, records, "patch_embed.proj.weight", rand((embed_dim, 3, 4, 4)))
    dump_tensor(out_dir, records, "patch_embed.proj.bias", zeros((embed_dim,)))
    dump_tensor(out_dir, records, "patch_embed.norm.weight", ones((embed_dim,)))
    dump_tensor(out_dir, records, "patch_embed.norm.bias", zeros((embed_dim,)))

    dim = embed_dim
    for stage, depth in enumerate(depths):
        for block in range(depth):
            prefix = f"layers.{stage}.blocks.{block}"
            dump_tensor(out_dir, records, f"{prefix}.norm1.weight", ones((dim,)))
            dump_tensor(out_dir, records, f"{prefix}.norm1.bias", zeros((dim,)))
            dump_tensor(out_dir, records, f"{prefix}.attn.qkv.weight", rand((3 * dim, dim)))
            dump_tensor(out_dir, records, f"{prefix}.attn.qkv.bias", zeros((3 * dim,)))
            dump_tensor(out_dir, records, f"{prefix}.attn.proj.weight", rand((dim, dim)))
            dump_tensor(out_dir, records, f"{prefix}.attn.proj.bias", zeros((dim,)))
            dump_tensor(out_dir, records, f"{prefix}.attn.relative_position_bias_table", rand(((2 * window - 1) * (2 * window - 1), heads[stage])))
            dump_tensor(out_dir, records, f"{prefix}.norm2.weight", ones((dim,)))
            dump_tensor(out_dir, records, f"{prefix}.norm2.bias", zeros((dim,)))
            dump_tensor(out_dir, records, f"{prefix}.mlp.fc1.weight", rand((4 * dim, dim)))
            dump_tensor(out_dir, records, f"{prefix}.mlp.fc1.bias", zeros((4 * dim,)))
            dump_tensor(out_dir, records, f"{prefix}.mlp.fc2.weight", rand((dim, 4 * dim)))
            dump_tensor(out_dir, records, f"{prefix}.mlp.fc2.bias", zeros((dim,)))
        if stage < 3:
            dump_tensor(out_dir, records, f"layers.{stage}.downsample.norm.weight", ones((4 * dim,)))
            dump_tensor(out_dir, records, f"layers.{stage}.downsample.norm.bias", zeros((4 * dim,)))
            dump_tensor(out_dir, records, f"layers.{stage}.downsample.reduction.weight", rand((2 * dim, 4 * dim)))
            dim *= 2

    dump_tensor(out_dir, records, "norm.weight", ones((dim,)))
    dump_tensor(out_dir, records, "norm.bias", zeros((dim,)))
    dump_tensor(out_dir, records, "head.weight", rand((num_classes, dim)))
    dump_tensor(out_dir, records, "head.bias", zeros((num_classes,)))

    meta = {
        "model": "swin_tiny_patch4_window7_224",
        "source": "TensorFlow/Keras synthetic deterministic export",
        "variant": "tiny",
        "dtype": "float32",
        "seed": seed,
        "tensors": records,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights/swin_tiny_patch4_window7_224")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    export_synthetic_swin_tiny(Path(args.output), args.seed)
    print(Path(args.output) / "metadata.json")


if __name__ == "__main__":
    main()
