# Swin TensorFlow Porting Map

This is an unofficial C inference port of Swin Transformer. The Microsoft PyTorch implementation remains architectural guidance, but the correctness reference for this project is the TensorFlow/Keras fixture path in `tools/swin_tf_reference.py`.

## Layout Policy

- Public C image input: NHWC `[B,H,W,C]`
- Internal token stream: NLC `[B,L,C]`
- Windows: `[B*nW, M*M, C]`
- Dense weights: C loader layout `[out,in]`; TensorFlow reference computes `x @ transpose(weight)`
- PatchEmbed weights: C loader layout `[out,in,kh,kw]`; TensorFlow reference transposes to `[kh,kw,in,out]` for `tf.nn.conv2d`
- LayerNorm epsilon: `1e-5`
- GELU: `tf.nn.gelu(approximate=False)`, equivalent to exact erf mode
- Attention mask values: `0.0` for same region, `-100.0` for different regions
- Relative position index order: TensorFlow/Numpy implementation follows Swin row-major meshgrid order with `index = (dh + M - 1) * (2M - 1) + (dw + M - 1)`

## Component Map

| TensorFlow reference | C target | Shape mapping | Notes |
| --- | --- | --- | --- |
| `swin_tf_reference.layer_norm` | `DM_SwinLayerNormBlock` | `[*,C] -> [*,C]` | epsilon fixed to `1e-5` |
| `swin_tf_reference.gelu` | local `gelu_inplace` | flat float32 | exact erf GELU |
| `swin_tf_reference.dense_c` | `DM_SwinLinearBlock` | `[B,L,in] -> [B,L,out]` | weight `[out,in]`, bias `[out]` |
| `swin_tf_reference.patch_embed_nhwc` | `DM_SwinPatchEmbedBlock` | `[B,H,W,3] -> [B,Hp*Wp,C]` | NHWC image, C-layout conv weights |
| `swin_tf_reference.window_partition` | `dm_swin_window_partition` | `[B,H,W,C] -> [B*nW,M,M,C]` | exact reshape/transpose order |
| `swin_tf_reference.window_reverse` | `dm_swin_window_reverse` | `[B*nW,M,M,C] -> [B,H,W,C]` | inverse of partition |
| `swin_tf_reference.relative_position_index` | `dm_swin_build_relative_position_index` | `[M*M,M*M] int32` | parity fixture generated for `M=7` |
| `swin_tf_reference.shifted_attention_mask` | `dm_swin_build_attention_mask` | `[nW,M*M,M*M]` | mask values `0/-100` |
| `swin_tf_reference.window_attention` | `dm_swin_window_attention_forward_fixture`, `DM_SwinWindowAttentionBlock` | `[B*nW,M*M,C] -> same` | QKV split order is `[q,k,v]` after dense `[3C]` |
| `swin_tf_reference.patch_merging` | `DM_SwinPatchMergingBlock` | `[B,H*W,C] -> [B,H/2*W/2,2C]` | concat order `x0,x1,x2,x3` |

## Pending Maps

TensorFlow fixture functions for full `DM_SwinBlock`, `DM_SwinStageBlock`, and `DM_SwinModel` logits still need to be added before claiming end-to-end Swin-T correctness.
