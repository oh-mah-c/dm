# dm — GGUF quantization & file-format core

Extracted from **ggml-org/llama.cpp** (https://github.com/ggml-org/llama.cpp).  
Only the two self-contained subsystems needed for **quantization** and **GGUF file I/O** were taken.
Everything else in llama.cpp (inference engine, ggml compute graph, backends, server) was left behind —
dm uses LibTorch + Vulkan for all tensor operations.

---

## What is here

### Quantization kernels — `src/ggml-quants.c` + `src/ggml-quants.h`

Pure C quantization math.  No dependency on the ggml compute graph or any backend.
Operates on raw `float *` buffers; call directly on LibTorch tensor `.data_ptr<float>()`.

Supported types:

| Type | Bits | Description |
|------|------|-------------|
| `Q4_0` | 4 | Absolute-value scale per 32-element block |
| `Q4_1` | 4 | Min+scale per 32-element block |
| `Q5_0` | 5 | 5-bit with abs scale |
| `Q5_1` | 5 | 5-bit with min+scale |
| `Q8_0` | 8 | Symmetric 8-bit (near-lossless) |
| `Q8_1` | 8 | Asymmetric 8-bit |
| `Q2_K` | 2 | Super-block k-quant |
| `Q3_K` | 3 | Super-block k-quant |
| `Q4_K` | 4 | Super-block k-quant (best 4-bit accuracy) |
| `Q5_K` | 5 | Super-block k-quant |
| `Q6_K` | 6 | Super-block k-quant (near-lossless) |
| `IQ2_XXS/XS/S/M` | ~2 | Importance-matrix weighted imatrix quants |
| `IQ3_XXS/S` | ~3 | imatrix quants |
| `IQ4_NL/XS` | ~4 | imatrix quants |
| `F16` | 16 | Half-precision passthrough |
| `BF16` | 16 | BFloat16 passthrough |

Key API pattern:
```c
// quantize a row of `k` floats → packed block_q4_K array
quantize_row_q4_K(const float *x, block_q4_K *y, int64_t k);

// dequantize back to float
dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k);

// high-level: pick type automatically, returns bytes written
ggml_quantize_chunk(ggml_type type, const float *src, void *dst,
                    int64_t start, int64_t nrows, int64_t n_per_row,
                    const float *imatrix);
```

### GGUF file format — `src/gguf.cpp` + `include/gguf.h`

Binary file format reader/writer.  Handles:
- Metadata KV header (architecture name, tokenizer vocab, hyperparameters, …)
- Tensor name/shape/type/offset directory
- Raw quantized tensor data blobs
- **Split files** (`model-00001-of-00003.gguf`) — writer emits a new part file when a size threshold is crossed
- **Merge** — open N part files, write one output (reverse of split)

Key API:
```c
// ── Write ──────────────────────────────────────────────────────────
struct gguf_context *ctx = gguf_init_empty();
gguf_set_val_str(ctx, "general.architecture", "llama");
gguf_set_val_u32(ctx, "llama.context_length", 4096);
gguf_add_tensor(ctx, tensor);          // ggml_tensor * with quantized data
gguf_write_to_file(ctx, "model.gguf", /*only_meta=*/false);
gguf_free(ctx);

// ── Read ───────────────────────────────────────────────────────────
struct gguf_init_params params = { .no_alloc = false, .ctx = &ggml_ctx };
struct gguf_context *ctx = gguf_init_from_file("model.gguf", params);
int idx = gguf_find_key(ctx, "general.architecture");
const char *arch = gguf_get_val_str(ctx, idx);
int t = gguf_find_tensor(ctx, "token_embd.weight");
size_t offset = gguf_get_tensor_offset(ctx, t);
gguf_free(ctx);
```

---

## Dependency map (only these files compile)

```
gguf.cpp
  ├── include/gguf.h          (public API)
  ├── include/ggml.h          (ggml_type enum, ggml_tensor struct, GGML_API macro)
  ├── include/ggml-backend.h  (ggml_backend_tensor_get — used for tensor data copy)
  └── src/ggml-impl.h         (internal helpers: ggml_fopen, ggml_init_params, …)

ggml-quants.c
  ├── src/ggml-quants.h       (block_q* structs, quantize/dequantize declarations)
  ├── src/ggml-common.h       (GGML_COMMON_DECL/IMPL macros, all block type defs)
  ├── src/ggml-impl.h         (GGML_ASSERT, fp16 helpers)
  └── src/ggml-cpu/ggml-cpu-impl.h  (GGML_RESTRICT, vec helpers used by quant loops)
```

Everything else in llama.cpp (`ggml.c`, backends, `llama.cpp`, server, examples) is **not included**
and is **not needed**.

---

## What was NOT taken (and why)

| llama.cpp path | Reason excluded |
|---|---|
| `ggml/src/ggml.c` + `ggml.cpp` | Compute graph engine — dm uses LibTorch |
| `ggml/src/ggml-backend*.c` | CPU/CUDA/Metal/Vulkan ggml backends — dm uses LibTorch + own Vulkan |
| `ggml/src/ggml-cpu/` (except `ggml-cpu-impl.h`) | CPU kernel implementations for ggml ops |
| `src/llama.cpp`, `src/llama.h` | Full inference engine with KV cache, sampling, context |
| `src/sampling.cpp`, `src/unicode*` | Sampling + Unicode — dm has its own tokenizer |
| `tools/`, `examples/` | CLI tools — dm will have its own `dm_gguf_quant` tool |
| `gguf-py/` | Python utilities — not needed in C++ dm core |
| `convert_*.py` | Python weight converters — dm exports directly from LibTorch |
