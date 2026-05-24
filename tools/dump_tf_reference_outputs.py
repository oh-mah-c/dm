#!/usr/bin/env python3
"""Generate TensorFlow/Keras golden fixtures for the C Swin port."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from swin_tf_reference import (
    dense_c,
    dump_bin,
    gelu,
    layer_norm,
    patch_embed_nhwc,
    patch_merging,
    relative_position_index,
    shifted_attention_mask,
    softmax_last_dim,
    window_attention,
    window_partition,
    window_reverse,
    write_metadata,
)


def rng(seed):
    return np.random.default_rng(seed)


def dump_all(out_dir: Path, seed: int):
    gen = rng(seed)
    records = []

    # LayerNorm
    x_ln = gen.normal(size=(2, 5, 8)).astype(np.float32)
    ln_w = gen.normal(size=(8,)).astype(np.float32)
    ln_b = gen.normal(size=(8,)).astype(np.float32)
    dump_bin(out_dir, records, "layernorm_input", x_ln)
    dump_bin(out_dir, records, "layernorm_weight", ln_w)
    dump_bin(out_dir, records, "layernorm_bias", ln_b)
    dump_bin(out_dir, records, "layernorm_expected", layer_norm(x_ln, ln_w, ln_b).numpy())

    # GELU
    x_gelu = gen.normal(size=(19,)).astype(np.float32)
    dump_bin(out_dir, records, "gelu_input", x_gelu)
    dump_bin(out_dir, records, "gelu_expected", gelu(x_gelu).numpy())

    # Dense
    x_dense = gen.normal(size=(2, 3, 5)).astype(np.float32)
    w_dense = gen.normal(size=(7, 5)).astype(np.float32)
    b_dense = gen.normal(size=(7,)).astype(np.float32)
    dump_bin(out_dir, records, "dense_input", x_dense)
    dump_bin(out_dir, records, "dense_weight", w_dense)
    dump_bin(out_dir, records, "dense_bias", b_dense)
    dump_bin(out_dir, records, "dense_expected", dense_c(x_dense, w_dense, b_dense).numpy())

    # Softmax
    x_soft = gen.normal(size=(4, 13)).astype(np.float32)
    dump_bin(out_dir, records, "softmax_input", x_soft)
    dump_bin(out_dir, records, "softmax_expected", softmax_last_dim(x_soft).numpy())

    # PatchEmbed
    patch = 4
    embed_dim = 6
    x_patch = gen.normal(size=(1, 8, 8, 3)).astype(np.float32)
    w_patch = gen.normal(size=(embed_dim, 3, patch, patch)).astype(np.float32)
    b_patch = gen.normal(size=(embed_dim,)).astype(np.float32)
    pe_ln_w = gen.normal(size=(embed_dim,)).astype(np.float32)
    pe_ln_b = gen.normal(size=(embed_dim,)).astype(np.float32)
    dump_bin(out_dir, records, "patch_embed_input", x_patch)
    dump_bin(out_dir, records, "patch_embed_weight", w_patch)
    dump_bin(out_dir, records, "patch_embed_bias", b_patch)
    dump_bin(out_dir, records, "patch_embed_norm_weight", pe_ln_w)
    dump_bin(out_dir, records, "patch_embed_norm_bias", pe_ln_b)
    dump_bin(out_dir, records, "patch_embed_expected", patch_embed_nhwc(x_patch, w_patch, b_patch, patch, pe_ln_w, pe_ln_b).numpy())

    # Window ops
    x_win = gen.normal(size=(1, 14, 14, 3)).astype(np.float32)
    wins = window_partition(x_win, 7).numpy()
    rev = window_reverse(wins, 7, 14, 14).numpy()
    dump_bin(out_dir, records, "window_partition_input", x_win)
    dump_bin(out_dir, records, "window_partition_expected", wins)
    dump_bin(out_dir, records, "window_reverse_expected", rev)

    # Required first milestone: relative position index, shifted mask, WindowAttention.
    rel_idx = relative_position_index(7)
    dump_bin(out_dir, records, "relative_position_index_window7", rel_idx, dtype="int32")

    mask = shifted_attention_mask(14, 14, 7, 3)
    dump_bin(out_dir, records, "attention_mask_h14_w14_window7_shift3", mask)

    dim = 12
    heads = 3
    n = 49
    x_attn = gen.normal(size=(2, n, dim)).astype(np.float32)
    qkv_w = gen.normal(size=(3 * dim, dim)).astype(np.float32)
    qkv_b = gen.normal(size=(3 * dim,)).astype(np.float32)
    proj_w = gen.normal(size=(dim, dim)).astype(np.float32)
    proj_b = gen.normal(size=(dim,)).astype(np.float32)
    rpb = gen.normal(size=((2 * 7 - 1) * (2 * 7 - 1), heads)).astype(np.float32)
    attn_out = window_attention(x_attn, qkv_w, qkv_b, proj_w, proj_b, rpb, heads, mask=None).numpy()
    shifted_attn_out = window_attention(x_attn, qkv_w, qkv_b, proj_w, proj_b, rpb, heads, mask=mask[:2]).numpy()
    dump_bin(out_dir, records, "window_attention_input", x_attn)
    dump_bin(out_dir, records, "window_attention_qkv_weight", qkv_w)
    dump_bin(out_dir, records, "window_attention_qkv_bias", qkv_b)
    dump_bin(out_dir, records, "window_attention_proj_weight", proj_w)
    dump_bin(out_dir, records, "window_attention_proj_bias", proj_b)
    dump_bin(out_dir, records, "window_attention_relative_position_bias_table", rpb)
    dump_bin(out_dir, records, "window_attention_expected", attn_out)
    dump_bin(out_dir, records, "window_attention_shifted_expected", shifted_attn_out)

    # PatchMerging
    pm_dim = 4
    x_pm = gen.normal(size=(1, 16, pm_dim)).astype(np.float32)
    pm_ln_w = gen.normal(size=(4 * pm_dim,)).astype(np.float32)
    pm_ln_b = gen.normal(size=(4 * pm_dim,)).astype(np.float32)
    pm_red = gen.normal(size=(2 * pm_dim, 4 * pm_dim)).astype(np.float32)
    dump_bin(out_dir, records, "patch_merging_input", x_pm)
    dump_bin(out_dir, records, "patch_merging_norm_weight", pm_ln_w)
    dump_bin(out_dir, records, "patch_merging_norm_bias", pm_ln_b)
    dump_bin(out_dir, records, "patch_merging_reduction_weight", pm_red)
    dump_bin(out_dir, records, "patch_merging_expected", patch_merging(x_pm, 4, 4, pm_ln_w, pm_ln_b, pm_red).numpy())

    # Placeholders for block/stage/full logits until C harness dumps matching intermediate outputs.
    # The primitive fixtures above are enough to begin TensorFlow parity debugging in the requested order.
    write_metadata(out_dir, records, {"seed": seed, "coverage": "primitive, window attention, patch merging"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="fixtures/swin")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dump_all(Path(args.output), args.seed)
    print(Path(args.output) / "metadata.json")


if __name__ == "__main__":
    main()
