# Swin TensorFlow Correctness Report

Correctness reference: TensorFlow/Keras fixture generation via `tools/swin_tf_reference.py` and `tools/dump_tf_reference_outputs.py`.

Required tolerance: `atol=1e-4`, `rtol=1e-4`.

Command run:

```bash
python3 tests/test_swin_tf_fixtures.py
```

| Test | Reference source | Max abs | Max rel | Status | Notes |
| --- | --- | ---: | ---: | --- | --- |
| Relative position index | `swin_tf_reference.relative_position_index` | 0 | 0 | pass | `M=7`, `2401` int32 values |
| Shifted attention mask | `swin_tf_reference.shifted_attention_mask` | 0 | 0 | pass | `H=14,W=14,M=7,shift=3`, values `0/-100` |
| WindowAttention | `swin_tf_reference.window_attention` | `3.8147e-05` | `3.43912e-06` | pass | dense weight layout `[out,in]`, QKV split `[q,k,v]` |

Generated fixtures currently include LayerNorm, GELU, Dense, PatchEmbed, window partition/reverse, relative position index, shifted attention mask, WindowAttention, and PatchMerging. C parity tests are currently wired for the first requested milestone: relative position index, shifted attention mask, and WindowAttention.

Do not claim full Swin-T correctness until W-MSA/SW-MSA block, stage, and full logits fixtures also pass within tolerance.
