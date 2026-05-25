# Installing dm from Scratch

This guide walks you through building **dm** on a fresh Ubuntu/Debian machine. dm uses **PyTorch (LibTorch)** as its ML core and **Vulkan-Hpp** as its GPU compute layer — both are included directly in the repo as native source code, so there are no submodules to init.

---

## What you will build

| Component | Where it ends up |
|---|---|
| LibTorch (PyTorch C++ library) | `src/core/pytorch/dist/` |
| Vulkan-Hpp headers | `src/core/gpu/Vulkan-Hpp/dist/` |
| dm binary | `build/dm` |
| dm_tokenizer binary | `build/dm_tokenizer` |

---

## Step 1 — Install system dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake ninja-build git \
    libvulkan-dev vulkan-headers glslc \
    libicu-dev \
    python3 python3-dev python3-numpy \
    libopenblas-dev
```

> **Why each package?**
> - `libvulkan-dev vulkan-headers` — Vulkan SDK for GPU compute
> - `glslc` — Vulkan shader compiler (separate from `glslang-tools`, must be this one)
> - `libicu-dev` — Unicode library used by the tokenizer
> - `python3-numpy` — needed by PyTorch's build system even when `BUILD_PYTHON=OFF`
> - `libopenblas-dev` — CPU BLAS backend for LibTorch

---

## Step 2 — Clone dm

```bash
git clone https://github.com/oh-mah-c/dm.git
cd dm
```

That's it — no `git submodule` commands. PyTorch and Vulkan-Hpp source code are already part of the repo.

---

## Step 3 — Build PyTorch (LibTorch)

This is the longest step. It compiles PyTorch as a pure C++ library (no Python runtime) with Vulkan GPU support.

```bash
cd src/core/pytorch
mkdir build && cd build

cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_CUDA=OFF \
    -DUSE_VULKAN=ON \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TEST=OFF \
    -DUSE_DISTRIBUTED=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_KINETO=OFF \
    -DGLSLC_EXECUTABLE=/usr/bin/glslc \
    -DCMAKE_INSTALL_PREFIX=../dist

cmake --build . --parallel 4   # use more cores if you have them
cmake --install .
```

> ⏱ This takes ~20–40 minutes depending on your machine. It compiles thousands of files.
>
> After `cmake --install .` you will see `src/core/pytorch/dist/lib/libtorch.so` and friends.

---

## Step 4 — Install Vulkan-Hpp headers

Vulkan-Hpp generates its C++ headers from the Vulkan spec at build time. After building the generator, you copy the headers into the dist folder.

```bash
# Go back to repo root first
cd ../../../../   # from src/core/pytorch/build

cd src/core/gpu/Vulkan-Hpp
mkdir build && cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Run the generator — this writes .hpp files into ../vulkan/
./VulkanHppGenerator

# Create the dist include dir
mkdir -p ../dist/include/vulkan
mkdir -p ../dist/include/vk_video

# Copy generated C++ headers
cp ../vulkan/*.hpp ../dist/include/vulkan/

# Copy C Vulkan headers from Vulkan-Headers subdir
cp ../Vulkan-Headers/include/vulkan/*.h ../dist/include/vulkan/

# Copy vk_video headers (needed by vulkan_core.h)
cp ../Vulkan-Headers/include/vk_video/* ../dist/include/vk_video/
```

---

## Step 5 — Build dm

Now build dm itself. From the repo root:

```bash
cd /path/to/dm   # make sure you are at repo root

mkdir build
cmake -S . -B build -G Ninja
cmake --build build --parallel 4
```

If everything worked you will see:

```
[235/235] Linking CXX executable dm
```

---

## Step 6 — Run

```bash
./build/dm
# Usage: dm <algo_id> <dataset_path> ...

./build/dm_tokenizer
```

---

## Troubleshooting

### `glslc not found` during PyTorch cmake
Make sure you installed `glslc` (not `glslang-tools`). Then pass it explicitly:
```bash
cmake .. -DGLSLC_EXECUTABLE=/usr/bin/glslc ...
```
If you already ran cmake once, delete the cache first: `rm CMakeCache.txt`

### `USE_VULKAN requires Vulkan installed`
```bash
sudo apt install libvulkan-dev vulkan-headers
```

### `unorm2_getNFKCInstance` linker error
You are missing `libicu-dev`:
```bash
sudo apt install libicu-dev
```

### PyTorch cmake fails with old cache
Always run cmake from a fresh `build/` directory, or delete `CMakeCache.txt` before re-running.

### `vulkan_enums.hpp not found` during Vulkan-Hpp install
You need to run `./VulkanHppGenerator` first (Step 4). The `.hpp` files are generated, not pre-committed.

### Headers not found when building dm
Make sure `cmake --install .` finished successfully for PyTorch (Step 3). The `dist/` folders must exist before building dm.

---

## Directory layout after full build

```
dm/
├── src/core/pytorch/
│   ├── (source — part of repo)
│   └── dist/            ← built by you in Step 3 (gitignored)
│       ├── lib/
│       │   ├── libtorch.so
│       │   ├── libtorch_cpu.so
│       │   └── ...
│       └── include/
├── src/core/gpu/Vulkan-Hpp/
│   ├── (source — part of repo)
│   └── dist/            ← built by you in Step 4 (gitignored)
│       └── include/
│           ├── vulkan/
│           └── vk_video/
└── build/               ← dm binaries (gitignored)
    ├── dm
    └── dm_tokenizer
```

The `dist/` and `build/` directories are in `.gitignore` — each developer builds them locally.
