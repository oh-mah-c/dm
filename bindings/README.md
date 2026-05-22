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
| § 5 | `Vision` — MobileNetV4-Tiny train/eval/predict |
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
