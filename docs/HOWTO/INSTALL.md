# dm — Install from Source

Two native cores: **PyTorch (LibTorch)** for ML + **Vulkan-Hpp** for GPU compute.
Both are built with **CMake**. No Bazel required.

---

## Requirements

| Tool | Version | Install |
|------|---------|---------|
| CMake | ≥ 3.18 | `sudo apt install cmake` |
| Ninja | any | `sudo apt install ninja-build` |
| Clang or GCC | 13+ | `sudo apt install clang` |
| Python 3 + NumPy | 3.10+ | `sudo apt install python3 python3-numpy` |
| OpenBLAS | any | `sudo apt install libopenblas-dev` |
| Vulkan SDK | 1.3+ | `sudo apt install libvulkan-dev glslang-tools` |
| Git | any | `sudo apt install git` |

### Ubuntu one-liner

```bash
sudo apt update && sudo apt install -y \
  cmake ninja-build clang \
  python3 python3-dev python3-numpy \
  libopenblas-dev \
  libvulkan-dev glslang-tools \
  git build-essential
```

---

## Clone

```bash
git clone <your-remote-url> dm
cd dm

# Submodules are already initialized if you followed the dev setup.
# If not:
git submodule update --init --recursive src/core/pytorch
git submodule update --init --recursive src/core/gpu/Vulkan-Hpp
```

---

## Phase 1 — Build PyTorch (LibTorch)

PyTorch lives at `src/core/pytorch/`. Building it with `BUILD_PYTHON=OFF`
produces **LibTorch** — a pure C++ shared library with no Python dependency
at runtime. This is the dm ML core.

### 1.1 — Configure

```bash
cd src/core/pytorch
mkdir -p build && cd build

cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DUSE_CUDA=OFF \
  -DUSE_ROCM=OFF \
  -DUSE_MPS=OFF \
  -DUSE_VULKAN=ON \
  -DUSE_OPENMP=ON \
  -DBUILD_PYTHON=OFF \
  -DBUILD_CAFFE2=OFF \
  -DBUILD_TEST=OFF \
  -DUSE_DISTRIBUTED=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_KINETO=OFF \
  -DCMAKE_INSTALL_PREFIX=../../engine/dist
```

> **`USE_VULKAN=ON`** — lets LibTorch tensors run Vulkan compute shaders,
> sharing the same GPU pipeline with Vulkan-Hpp.

> **`USE_OPENMP=ON`** — CPU parallelism for free. Remove if you don't have
> `libgomp` installed.

### 1.2 — Build

```bash
# Still inside src/core/pytorch/build/
# Rule of thumb: 1 job per 3 GB of free RAM
# 8 GB  → --parallel 2
# 16 GB → --parallel 4
# 32 GB → --parallel 8

cmake --build . --parallel 4
```

> **First build takes 30–90 minutes** depending on CPU and RAM.
> Subsequent incremental builds are fast.

### 1.3 — Install

```bash
cmake --install .
```

Artifacts land in `src/core/engine/dist/`:

```
src/core/engine/dist/
  include/
    torch/          ← torch/torch.h, ATen/, c10/
  lib/
    libtorch_cpu.so
    libc10.so
    libgomp.so.1
  share/
    cmake/Torch/    ← TorchConfig.cmake  (CMake find_package target)
```

---

## Phase 2 — Build Vulkan-Hpp

Vulkan-Hpp lives at `src/core/gpu/Vulkan-Hpp/`. It is **header-only** at the
dm level — you don't need to build samples or tests. The only thing needed is
the `Vulkan-Headers` submodule (C Vulkan headers) which was already fetched.

```bash
cd src/core/gpu/Vulkan-Hpp
mkdir -p build && cd build

cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVULKAN_HPP_BUILD_TESTS=OFF \
  -DVULKAN_HPP_BUILD_SAMPLES=OFF \
  -DVULKAN_HPP_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX=../dist
```

```bash
cmake --build . --parallel 4
cmake --install .
```

Headers install to `src/core/gpu/Vulkan-Hpp/dist/include/vulkan/`.

> If you only need headers (no install), skip the build entirely —
> just point `#include` at `src/core/gpu/Vulkan-Hpp/vulkan/vulkan.hpp` directly.

---

## Phase 3 — Build dm

From the **dm root**:

```bash
cd /path/to/dm
mkdir -p build && cd build

cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="src/core/engine/dist;src/core/gpu/Vulkan-Hpp/dist"

cmake --build . --parallel 4
```

---

## Quick reference — full sequence

```bash
# 1. Clone + submodules
git clone <url> dm && cd dm
git submodule update --init --recursive src/core/pytorch
git submodule update --init --recursive src/core/gpu/Vulkan-Hpp

# 2. System deps
sudo apt install -y cmake ninja-build clang python3 python3-dev \
  python3-numpy libopenblas-dev libvulkan-dev glslang-tools build-essential

# 3. Build PyTorch → installs to src/core/engine/dist/
cd src/core/pytorch && mkdir -p build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DUSE_CUDA=OFF -DUSE_VULKAN=ON \
  -DBUILD_PYTHON=OFF -DBUILD_CAFFE2=OFF -DBUILD_TEST=OFF \
  -DUSE_DISTRIBUTED=OFF -DUSE_NCCL=OFF -DUSE_KINETO=OFF \
  -DCMAKE_INSTALL_PREFIX=../../engine/dist
cmake --build . --parallel 4 && cmake --install .
cd ../../..

# 4. Vulkan-Hpp (header-only — just init submodule, no build needed)
# Already done via: git submodule update --init --recursive src/core/gpu/Vulkan-Hpp

# 5. Build dm
mkdir -p build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="src/core/engine/dist;src/core/gpu/Vulkan-Hpp"
cmake --build . --parallel 4
```

---

## Subsequent builds

PyTorch only needs rebuilding if you change files inside `src/core/pytorch/`.
For all dm changes just run from the dm root:

```bash
cd build && cmake --build . --parallel 4
```

CMake tracks dependencies — only changed targets recompile.

---

## Troubleshooting

### OOM during PyTorch build
Reduce parallel jobs. Each job uses ~2–3 GB RAM:
```bash
cmake --build . --parallel 2   # 6 GB needed
cmake --build . --parallel 1   # 3 GB needed (slow but safe)
```

### `Could not find Torch` when building dm
PyTorch install step was skipped or `--prefix` path is wrong.
Check:
```bash
ls src/core/engine/dist/share/cmake/Torch/TorchConfig.cmake
```
If missing, re-run `cmake --install .` from `src/core/pytorch/build/`.

### `vulkan/vulkan.h: No such file or directory`
`Vulkan-Headers` submodule is empty. Run:
```bash
git submodule update --init src/core/gpu/Vulkan-Hpp/Vulkan-Headers
```

### `undefined reference to omp_get_thread_num`
OpenMP not installed. Either install `libgomp`:
```bash
sudo apt install libgomp1
```
Or rebuild PyTorch with `-DUSE_OPENMP=OFF`.

### `<atomic> file not found` or stdlib conflicts
System libc++ is missing. Use libstdc++ (default on Ubuntu):
```bash
cmake .. -DCMAKE_CXX_FLAGS="-stdlib=libstdc++"
```

### PyTorch configure step fails on `No module named numpy`
```bash
pip3 install numpy
```
