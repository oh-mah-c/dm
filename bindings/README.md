# DM Framework — Language Bindings

All bindings are generated from the single stable C ABI defined in
[`include/dm.h`](../include/dm.h).  Build **libdm** first, then pick
the language that fits your project.

---

## Table of contents

| Language | Location | Mechanism |
|----------|----------|-----------|
| **C**    | `include/dm.h` | Direct (libdm *is* a C library) |
| **C++**  | `bindings/cpp/dm.hpp` | Header-only RAII wrappers |
| **Python** | `bindings/python/dm/` | `ctypes` (no compilation) |
| **Go**   | `bindings/go/dm/` | CGo |
| **JavaScript** | `bindings/js/dm/` | `ffi-napi` (Node.js) |
| **Java** | `bindings/java/com/dm/` | JNA |

---

## Build libdm

```bash
# Linux / macOS (GNU Make)
make -C /path/to/dm   # produces libdm.so (Linux) or libdm.dylib (macOS)

# Windows (MSVC, see build.ps1)
pwsh build.ps1
```

The shared library must be discoverable by the target runtime. The
easiest approach during development is:

```bash
export LD_LIBRARY_PATH=/path/to/dm        # Linux
export DYLD_LIBRARY_PATH=/path/to/dm      # macOS
set PATH=%PATH%;C:\path\to\dm             # Windows
```

Or set the language-specific env var listed below.

---

## C

`dm.h` is the binding — no wrappers needed.

```c
#include "dm.h"

int main(void) {
    dm_init();
    printf("%s\n", dm_version());
    return 0;
}
```

**Compile:**
```bash
gcc -std=c11 -I/path/to/dm/include main.c -L/path/to/dm -ldm -o app
```

**Example:** [`examples/c/example.c`](../examples/c/example.c)

---

## C++ (header-only)

```cpp
#include "bindings/cpp/dm.hpp"

int main() {
    dm::init();
    std::cout << dm::version() << "\n";

    dm::Algorithm algo("fpgrowth");
    algo.run("data.txt", "out.txt", 0.05);
}
```

**Compile:**
```bash
g++ -std=c++17 \
    -I/path/to/dm/include \
    -I/path/to/dm/bindings/cpp \
    main.cpp -L/path/to/dm -ldm -o app
```

**Example:** [`examples/cpp/example.cpp`](../examples/cpp/example.cpp)

---

## Python

**Install:**
```bash
pip install -e bindings/python        # editable install
# or just add bindings/python to PYTHONPATH

export DM_LIB=/path/to/libdm.so      # or .dylib / .dll
```

```python
import dm

dm.init()
print(dm.version())

with dm.Algorithm("fpgrowth") as algo:
    algo.run("mushrooms.txt", "/tmp/out.txt", 0.05)

with dm.Tokenizer("bpe") as tok:
    tok.train("corpus.txt", 2000, "/tmp/model")
    ids = tok.encode("hello world")
    print(tok.decode(ids))
```

**Requirements:** Python ≥ 3.9, no extra packages (pure `ctypes`).

**Example:** [`examples/python/example.py`](../examples/python/example.py)

---

## Go

**Requirements:** Go ≥ 1.21, CGo enabled, a C compiler.

```bash
# In your module, add the replace directive:
# require github.com/pomaieco/dm v0.0.0
# replace github.com/pomaieco/dm => /path/to/dm/bindings/go

CGO_CFLAGS="-I/path/to/dm/include" \
CGO_LDFLAGS="-L/path/to/dm -ldm" \
go build ./...
```

```go
import dm "github.com/pomaieco/dm/dm"

dm.Init()
fmt.Println(dm.Version())

algo, _ := dm.NewAlgorithm("fpgrowth")
defer algo.Close()
algo.Run("mushrooms.txt", "/tmp/out.txt", 0.05, nil)
```

**Example:** [`examples/go/main.go`](../examples/go/main.go)

---

## JavaScript (Node.js)

**Requirements:** Node.js ≥ 14, native addons build tools.

```bash
cd bindings/js/dm
npm install ffi-napi ref-napi ref-array-di ref-struct-di

export DM_LIB=/path/to/libdm.so
```

```js
const dm = require('./bindings/js/dm');

dm.init();
console.log(dm.version());

const algo = new dm.Algorithm('fpgrowth');
algo.run('mushrooms.txt', '/tmp/out.txt', 0.05);
algo.close();

const tok = new dm.Tokenizer('bpe');
tok.train('corpus.txt', 2000, '/tmp/model');
const ids = tok.encode('hello world');
console.log(tok.decode(ids));
tok.close();
```

**Example:** [`examples/js/example.js`](../examples/js/example.js)

---

## Java (JNA)

**Requirements:** Java ≥ 11, JNA 5.x.

```xml
<!-- Maven -->
<dependency>
  <groupId>net.java.dev.jna</groupId>
  <artifactId>jna</artifactId>
  <version>5.14.0</version>
</dependency>
```

```bash
# Compile
javac -cp jna-5.14.0.jar:bindings/java bindings/java/com/dm/DM.java

# Run
java -Djna.library.path=/path/to/dm \
     -cp .:jna-5.14.0.jar:bindings/java \
     YourApp
```

```java
import com.dm.DM;

DM.init();
System.out.println(DM.version());

try (DM.Algorithm algo = new DM.Algorithm("fpgrowth")) {
    algo.run("mushrooms.txt", "/tmp/out.txt", 0.05);
}

try (DM.Tokenizer tok = new DM.Tokenizer("bpe")) {
    tok.train("corpus.txt", 2000, "/tmp/model");
    int[] ids = tok.encode("hello world");
    System.out.println(tok.decode(ids));
}
```

**Environment variable:** `DM_LIB=/path/to/libdm.so` (fallback: `-Djna.library.path=…`)

**Example:** [`examples/java/ExampleApp.java`](../examples/java/ExampleApp.java)

---

## dm_engine — Building custom models

`dm_engine` is the neural-compute core of dm.  Every op dispatches through
the **TensorFlow Eager C API** (`TFE_*`), so XLA, cuDNN, and oneDNN
acceleration is available automatically — no extra configuration.

Users can compose custom architectures from these primitives exactly the same
way they would compose TensorFlow layers:

```python
# Python
import dm

x = dm.Tensor(1, 3, 224, 224)   # NCHW batch=1, RGB 224×224
y = dm.Tensor(1, 64, 112, 112)
dm.op.conv2d_same(x, y, w, b, 64, 3, 2)
dm.op.relu(y)
x.free(); y.free()   # or use as context manager
```

```cpp
// C++
dm::Tensor x(1, 3, 224, 224), y(1, 64, 112, 112);
dm::op::conv2d_same(x, y, w, b, 64, 3, 2);
dm::op::relu(y);
```

```go
// Go
x, _ := dm.NewTensor(1, 3, 224, 224)
y, _ := dm.NewTensor(1, 64, 112, 112)
defer x.Free(); defer y.Free()
dm.OpConv2dSame(x, y, w, b, 64, 3, 2)
dm.OpRelu(y)
```

```js
// JavaScript
const x = new dm.Tensor(1, 3, 224, 224);
const y = new dm.Tensor(1, 64, 112, 112);
dm.op.conv2dSame(x, y, w, b, 64, 3, 2);
dm.op.relu(y);
x.free(); y.free();
```

```java
// Java
try (DM.Tensor x = new DM.Tensor(1, 3, 224, 224);
     DM.Tensor y = new DM.Tensor(1, 64, 112, 112)) {
    DM.Op.conv2dSame(x, y, w, b, 64, 3, 2);
    DM.Op.relu(y);
}
```

### Available primitives

| Category | Operations |
|----------|------------|
| **Tensor** | `alloc`, `free`, `fill`, `get`, `set`, `count` |
| **Convolutions** | `conv2d_same` (OIHW), `depthwise_conv` ([c][k][k]), `pointwise_conv` (1×1) |
| **Linear** | `linear` — [n,in,1,1] → [n,out,1,1] |
| **Pooling** | `global_avg_pool`, `max_pool2d_same` |
| **Normalisation** | `batch_norm` (TFE FusedBatchNorm), `layer_norm` (seq models) |
| **Elementwise** | `add` (residual connections) |
| **Activations** | `relu`, `relu6`, `tanh`, `sigmoid`, `gelu` |
| **Softmax** | `softmax` (tensor), `softmax_rows` (raw buffer) |
| **MatMul** | `matmul_nt` (A×Bᵀ), `matmul_nn` (A×B) |
| **Backward** | `linear_backward`, `relu_backward`, `tanh_backward` |
| **Maxout** | `maxout` + `maxout_backward` |
| **Dropout** | `dropout` + `dropout_backward` |
| **Optimisers** | `adam_step`, `adagrad_step`, `sgd_momentum_step` |

All ops dispatch through `TFE_*` (backed by TF C++ runtime) except the
optimiser steps and backward passes, which are pure-C (no TFE needed).

**Tensor layout:** NCHW (`n, c, h, w`), row-major, contiguous `float32`.

---

## API surface covered by all bindings

| Section | Feature |
|---------|---------|
| § 1 | `version()`, `init()`, `strerror()` |
| § 2 | `Dataset` — open, count, max_id |
| § 3 | `Algorithm` — 132 algorithms, run, list |
| § 4 | `Tokenizer` — train, load, encode, decode, VOLT |
| § 5 | `Vision` — MobileNetV4-Tiny train/eval/predict + **TinyViT-5M/11M/21M** |
| § 6 | `LM` — **Transformer** (Vaswani et al. 2017) enc-dec, TinyTransformer, TinyStories, and **BERT encoder inference** (`model_type="bert"`; space-separated token IDs in, pooled `[CLS]` values out). BERT follows Devlin, Chang, Lee, and Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL-HLT 2019, ACL Anthology N19-1423. |
| § 7 | Image load/resize/patchify (via C API directly) |
| § 8 | **Engine** — `Tensor` + full neural-op suite: convolutions, linear, pooling, batch/layer norm, activations (relu/relu6/tanh/sigmoid/gelu), softmax, matmul, backward passes, maxout, dropout, Adam/Adagrad/SGD optimisers |
| § 9 | `Benchmark` — phase timing, report, print |
| § 10 | `BitSet` — set/clear/get/and/or/not/popcount |
| § 11 | `DataGen` — MEDM synthetic, Textbook corpus |
| § 12 | Plugin load/list/run (via `cliRun`) |
| § 13 | `cliRun` — any dm sub-command |
| § 14 | `GpuCtx` — Vulkan BPE/Sinkhorn/Unigram EM |
| § 15 | `Arena` (C++ wrapper; C struct elsewhere) |
| § 16 | `MMap` (C direct use; C++ wrapper) |
| § 17 | `FlatDataset` / Connector (C direct use) |
| § 18 | Experiment helpers — timer, peak_ram, CSV log |

---

---

## TinyViT — Fast Pretraining Distillation

**Paper:** Wu, Zhang, Peng et al., *TinyViT: Fast Pretraining Distillation for
Small Vision Transformers*, arXiv:2207.10666v1 (2022)

TinyViT is a family of tiny vision transformers trained with a fast
knowledge-distillation framework. Three variants are available:

| Variant | Parameters | Embed dims (D1,D2,D3,D4) |
|---------|-----------|--------------------------|
| `tinyvit_5m`  |  ~5 M  | {64,  128, 160, 320} |
| `tinyvit_11m` | ~11 M  | {64,  128, 256, 448} |
| `tinyvit_21m` | ~21 M  | {96,  192, 384, 576} |

All variants share: depths={2,2,6,2}, windows={7,14,7}, MBConv-R=4, MLP-M=4,
head-dim E=32.

### Architecture

```
Input 224×224×3
 └─ Patch Embed ─ 2× Conv3×3(stride 2, pad 1, BN+GELU) ──→ 56×56×D1
 └─ Stage 1 ─────  2× MBConv(D1, stride=1)              ──→ 56×56×D1
 └─ Downsample ──  MBConv(D1→D2, stride=2)              ──→ 28×28×D2
 └─ Stage 2 ─────  2× Transformer(window 7×7)           ──→ 28×28×D2
 └─ Downsample ──  MBConv(D2→D3, stride=2)              ──→ 14×14×D3
 └─ Stage 3 ─────  6× Transformer(window 14×14)         ──→ 14×14×D3
 └─ Downsample ──  MBConv(D3→D4, stride=2)              ──→  7× 7×D4
 └─ Stage 4 ─────  2× Transformer(window 7×7)           ──→  7× 7×D4
 └─ Head ─────────  AvgPool + LayerNorm + Linear         ──→ num_classes

Transformer block:
  LayerNorm → Window-MHSA(rel-pos biases) → +residual
  DW-Conv3×3 local mixer                  → +residual
  LayerNorm → MLP(GELU)                   → +residual
```

### Fast Pretraining Distillation (§ 3.1)

The paper's key contribution: instead of running the large teacher model at
every training step, teacher soft-labels are **pre-computed once** and stored
on disk as sparse top-K logits.

```
Store per image:  { top-K indices, top-K values, aug_seed }  (Eq. 2)
Train student:    L = CE(ŷ_teacher_recovered, S(student_logits))
```

This reduces memory and allows the student to train at full batch size with
no teacher GPU cost at runtime.

### CLI usage

```bash
# Pure-C inference (random weights — train first for real results)
dm tinyvit infer -i cat.ppm --variant 21m --classes 1000

# TF/Keras training from scratch
dm tinyvit train --manifest train.txt -o weights.bin \
    --variant 21m --classes 1000 --epochs 90 --batch 256 --lr 0.002

# Fast distillation training using stored sparse labels
dm tinyvit distill --manifest train.txt --labels teacher_labels.bin \
    -o weights.bin --variant 21m --K 100 --epochs 90

# Generate sparse teacher labels from a SavedModel
dm tinyvit gen-labels --teacher /path/to/teacher_saved_model \
    --manifest imgs.txt -o labels.bin --K 100 --classes 21841

# Benchmark forward-pass throughput
dm tinyvit bench --variant 21m --batch 1
```

### Using the C API directly (§ 5)

```c
#include "dm.h"

/* 1. Query weight count and allocate */
size_t wc = dm_tinyvit_weight_count(DM_TINYVIT_21M, 1000, 224);
float *weights = calloc(wc, sizeof(float));

/* 2. Load trained weights (produced by 'dm tinyvit train') */
DM_TinyViTVariant var; int classes, img_size;
dm_tinyvit_load("weights.bin", &var, &classes, &img_size, &weights);

/* 3. Forward pass (NHWC float input, values in [0,1]) */
float logits[1000];
dm_tinyvit_forward(DM_TINYVIT_21M, weights,
                   input_nhwc, /*batch=*/1,
                   1000, 224, logits);

/* 4. Distillation loss against stored sparse label */
float loss;
dm_tinyvit_distill_loss(student_logits,
                         label_indices,   /* uint32[K] */
                         label_values,    /* float[K]  */
                         /*K=*/100, /*C=*/21841,
                         /*temperature=*/1.0f, &loss);
free(weights);
```

### Python (ctypes)

```python
import dm, ctypes

dm.init()

# Via Vision handle (train/eval)
with dm.Vision("tinyvit_21m") as v:
    v.train("train.txt", epochs=90, batch_size=256, lr=2e-3)

# Direct C inference
wc = dm.lib.dm_tinyvit_weight_count(2, 1000, 224)  # variant=2 → 21M
weights = (ctypes.c_float * wc)()
dm.lib.dm_tinyvit_load(b"weights.bin", None, None, None,
                        ctypes.byref(ctypes.cast(weights, ctypes.POINTER(ctypes.c_float))))
logits = (ctypes.c_float * 1000)()
dm.lib.dm_tinyvit_forward(2, weights, input_nhwc, 1, 1000, 224, logits)
```

### Fast Distillation — Sparse Label Generation (any language)

```bash
# Step 1: Generate labels once (requires teacher SavedModel)
dm tinyvit gen-labels \
    --teacher swin_l_saved_model/ \
    --manifest imagenet21k.txt \
    -o imagenet21k_labels_K100.bin \
    --K 100 --classes 21841

# Step 2: Train student repeatedly (no teacher needed at all)
dm tinyvit distill \
    --manifest imagenet21k.txt \
    --labels imagenet21k_labels_K100.bin \
    -o tinyvit21m.bin \
    --variant 21m --epochs 90
```

---

## Transformer — "Attention Is All You Need"

**Paper:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin,
*Attention Is All You Need*, NeurIPS 2017 (arXiv:1706.03762)

The Transformer is the foundational encoder-decoder sequence transduction model
based entirely on attention mechanisms.  Every building block is exposed as a
standalone reusable function.

### Architecture

```
Input tokens  →  Embedding (×√d_model) + Sinusoidal PE
  └─ Encoder: N=6 layers, each:
        Self-MHA  →  Add & LayerNorm
        FFN       →  Add & LayerNorm
  └─ Final encoder LayerNorm

Output tokens (shifted right)  →  Embedding + PE
  └─ Decoder: N=6 layers, each:
        Masked Self-MHA  →  Add & LayerNorm
        Cross-MHA        →  Add & LayerNorm   (Q=decoder, K=V=encoder output)
        FFN              →  Add & LayerNorm
  └─ Linear projection (shared with embedding) →  Softmax → Probabilities
```

**Shared weight matrix** (§3.4): source embedding, target embedding, and
pre-softmax linear projection all use the same weight matrix (transposed for
projection), scaled by √d_model when applied.

### Hyperparameters (Table 3)

| Variant | N | d_model | d_ff | h | d_k=d_v | P_drop | Params |
|---------|---|---------|------|---|---------|--------|--------|
| `base`  | 6 | 512  | 2048 | 8  | 64 | 0.1 | ~65 M  |
| `big`   | 6 | 1024 | 4096 | 16 | 64 | 0.3 | ~213 M |

### Reusable Primitives (§3)

Each module is independently callable for use in other architectures:

| Function | Paper reference |
|----------|----------------|
| `dm_transformer_positional_encoding` | §3.5 — sin/cos encoding |
| `dm_transformer_layer_norm` | §3.1 — Ba et al. 2016 |
| `dm_transformer_sdp_attention` | §3.2.1 Eq. 1 — Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V |
| `dm_transformer_causal_mask` | §3.2.3 — upper-triangle −∞ mask |
| `dm_transformer_mha` | §3.2.2 Eq. 2 — MultiHead(Q,K,V)=Concat(heads)Wᴼ |
| `dm_transformer_ffn` | §3.3 Eq. 2 — FFN(x)=max(0,xW₁+b₁)W₂+b₂ |
| `dm_transformer_encoder_layer` | §3.1 — single encoder layer |
| `dm_transformer_decoder_layer` | §3.1 — single decoder layer |
| `dm_transformer_encode` | §3.1 — full encoder stack |
| `dm_transformer_decode` | §3.1 — full decoder stack + output projection |
| `dm_transformer_forward` | §3 — full encoder-decoder forward pass |
| `dm_transformer_lr_schedule` | §5.3 Eq. 3 — Adam warmup schedule |

### CLI usage

```bash
# Train (requires TensorFlow/Keras — exact paper setup)
dm transformer train \
    --src train.src.tok  --tgt train.tgt.tok \
    -o transformer_base.bin \
    --variant base --vocab-size 37000 --max-len 512 \
    --epochs 100 --batch 32 --warmup 4000

# Greedy decode (pure-C inference, no TF needed)
dm transformer infer -m transformer_base.bin \
    --src "10 20 30 40 50" --max-tokens 64

# Encoder only (export contextual representations)
dm transformer encode --src tokens.txt -m transformer_base.bin -o enc.bin

# Throughput benchmark (pure-C, zero-initialized weights)
dm transformer bench --variant base --src-len 64 --tgt-len 64
dm transformer bench --variant big  --src-len 32 --tgt-len 32

# Show model metadata
dm transformer info -m transformer_base.bin
```

### C API usage

```c
#include "dm.h"              /* or directly: #include "models/lm/transformer.h" */

/* 1. Configure */
TransformerConfig cfg;
dm_transformer_config_init(&cfg, TRANSFORMER_BASE, 37000, 512);

/* 2. Count + allocate weights */
size_t wc = dm_transformer_weight_count(&cfg);
float *weights = malloc(wc * sizeof(float));
dm_transformer_load("transformer_base.bin", &cfg, &weights);

/* 3. Full forward pass */
int src[] = {10, 20, 30};
int tgt[] = {1, 50};               /* 1 = BOS */
float logits[2 * 37000];
dm_transformer_forward(&cfg, weights, src, 3, tgt, 2, logits);

/* 4. Use individual modules in your own architecture */
float pe[512 * 512];
dm_transformer_positional_encoding(512, 512, pe);

float out[seq * 512];
dm_transformer_encoder_layer(x_in, &cfg, layer_weights, seq, out);

/* 5. Learning rate at step t (§5.3 Eq. 3) */
float lr = dm_transformer_lr_schedule(512, /*step=*/1000, /*warmup=*/4000);

free(weights);
```

### Python usage

```python
import dm

dm.init()

# Train via CLI passthrough
dm.cliRun("transformer", ["train",
    "--src", "train.src.tok", "--tgt", "train.tgt.tok",
    "-o", "model.bin", "--variant", "base",
    "--vocab-size", "37000", "--epochs", "100"])

# Infer
dm.cliRun("transformer", ["infer", "-m", "model.bin",
    "--src", "10 20 30", "--max-tokens", "64"])
```

### Training details (§5)

- **Optimizer:** Adam, β₁=0.9, β₂=0.98, ε=10⁻⁹
- **LR schedule (Eq. 3):** `lrate = d_model⁻⁰·⁵ · min(step⁻⁰·⁵, step · warmup⁻¹·⁵)`
  - Linear warmup for first `warmup_steps=4000` steps, then ∝ step⁻⁰·⁵ decay
- **Residual dropout** (P_drop=0.1 base / 0.3 big): applied after each sub-layer
  before Add & Norm, and on embedding + PE sums
- **Label smoothing** ε_ls=0.1 (§5.4): improves accuracy / BLEU despite hurting
  perplexity
- **Shared embedding weight** (§3.4): reduces parameters and improves quality

---

## Examples directory

```
examples/
├── c/          example.c        — pure C demo
├── cpp/        example.cpp      — C++ RAII demo
├── python/     example.py       — Python demo
├── go/         main.go          — Go demo
├── js/         example.js       — Node.js demo
└── java/       ExampleApp.java  — Java demo
```
