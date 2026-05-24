# Swin Weight Format

`tools/export_swin_tf_weights.py` writes:

```text
weights/swin_tiny_patch4_window7_224/
  metadata.json
  tensors/
    patch_embed.proj.weight.bin
    ...
```

Each `.bin` tensor is raw little-endian float32 in the C loader layout used by the TensorFlow reference tools. `metadata.json` records `name`, `shape`, `dtype`, and `file` for every tensor. The C loader validates required tensor names and exact shapes, reports missing tensors, and reports unused tensors.

No tensor is silently transposed. The C implementation documents and handles layout at the operator boundary:

- image input: NHWC
- token stream: NLC
- windows: `[B*nW, M*M, C]`
- patch projection weight: `[out,in,kh,kw]`; TensorFlow reference transposes to `[kh,kw,in,out]`
- dense weights: official `[out,in]`
- relative position bias table: official `[(2M-1)*(2M-1), num_heads]`
