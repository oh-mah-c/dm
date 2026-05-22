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

## API surface covered by all bindings

| Section | Feature |
|---------|---------|
| § 1 | `version()`, `init()`, `strerror()` |
| § 2 | `Dataset` — open, count, max_id |
| § 3 | `Algorithm` — 132 algorithms, run, list |
| § 4 | `Tokenizer` — train, load, encode, decode, VOLT |
| § 5 | `Vision` — MobileNetV4-Tiny train/eval/predict + **TinyViT-5M/11M/21M** |
| § 6 | `LM` — Tiny Transformer / TinyStories generate |
| § 7 | Image load/resize/patchify (via C API directly) |
| § 8 | `Tensor` + neural ops (C++ wrapper; raw C elsewhere) |
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
