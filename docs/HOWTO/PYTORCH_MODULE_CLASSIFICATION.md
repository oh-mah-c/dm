# PyTorch Module Classification

This document classifies every major module/folder in `src/core/pytorch/` to guide a future
minimal CPU-only LibTorch build. **No source is deleted based on this document.** Modules
classified as DEFERRED_IMPORTANT or OPTIONAL_BACKEND are excluded from the CPU-only build
via CMake flags — the source snapshot remains intact.

## Classification Key

| Class | Meaning |
|---|---|
| **CORE_NOW** | Required for milestone 1: CPU tensor ops, autograd, nn, optimizers, save/load |
| **DEFERRED_IMPORTANT** | Not needed for milestone 1 but must be preserved — ONNX, quantization, JIT, graph export |
| **OPTIONAL_BACKEND** | Hardware backends not needed on CPU-only: CUDA, ROCm, MPS, Vulkan, distributed |
| **ARCHIVE_ONLY** | Docs, tests, benchmarks, CI scripts — never compiled, kept as reference |
| **REMOVE_ONLY_IF_SAFE** | Generated files and build artifacts only — fully regenerable, safe to clean |

---

## CPU-only CMake flags (milestone 1)

```bash
cd src/core/pytorch && mkdir -p build_cpu && cd build_cpu
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF -DATEN_NO_TEST=ON \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF -DUSE_MPS=OFF \
  -DUSE_VULKAN=OFF -DUSE_DISTRIBUTED=OFF -DUSE_NCCL=OFF \
  -DUSE_KINETO=OFF -DUSE_FBGEMM=ON -DUSE_OPENMP=ON \
  -DUSE_MKLDNN=ON -DCMAKE_INSTALL_PREFIX=../dist
cmake --build . --parallel $(nproc)
cmake --install .
```

---

## Top-level source directories

| Path | Class | Reason | Build status (current) | Future usefulness | Risk if removed | How to re-enable |
|---|---|---|---|---|---|---|
| `c10/` | CORE_NOW | Foundational tensor metadata, Device, Stream, Allocator — all libraries depend on it | Always built | Critical forever | Breaks everything | N/A — always on |
| `aten/` | CORE_NOW | ATen tensor library: TensorImpl, dispatch, CPU kernels, native ops | Built via INTERN_BUILD_ATEN_OPS | Core math forever | Breaks all tensor ops | N/A — always on |
| `caffe2/core/` | CORE_NOW | Operator registry, workspace, blob — links ATen into libtorch_cpu | Always built | Required by libtorch_cpu | Breaks library | N/A |
| `caffe2/serialize/` | CORE_NOW | `.pt` model save/load (minimal serialization) | Always built | Checkpoint I/O | Breaks save/load | N/A |
| `torch/csrc/api/` | CORE_NOW | C++ `nn::Module`, `Parameter`, `Sequential`, `Linear`, SGD/Adam/AdamW | Always built | Primary C++ training API | Breaks C++ training | N/A |
| `torch/csrc/autograd/` | CORE_NOW | Autograd engine, `Variable`, `grad_fn`, backward graph | Always built | Backprop forever | Breaks training | N/A |
| `torchgen/` | CORE_NOW | Build-time codegen for operator dispatch tables | Build-time Python tool | Must re-run to rebuild | Breaks operator codegen | N/A — run at build time |
| `torch/csrc/jit/` | DEFERRED_IMPORTANT | TorchScript compiler, graph IR, serialization of scripted models | Always built in libtorch_cpu | Graph export, static dispatch, model portability | None — source preserved | Already compiled; prune later with `USE_LIGHTWEIGHT_DISPATCH=ON` |
| `torch/csrc/onnx/` | DEFERRED_IMPORTANT | ONNX export glue (Python-facing) | Built when `BUILD_PYTHON=ON` | Export models to ONNX runtime | None — source preserved | Add `BUILD_PYTHON=ON` |
| `third_party/onnx/` | DEFERRED_IMPORTANT | ONNX schema, proto definitions | Built when `ONNX_ML=ON` | Required for ONNX import/export | None — source preserved | `ONNX_ML=ON USE_SYSTEM_ONNX=OFF` |
| `torch/csrc/export/` | DEFERRED_IMPORTANT | `torch.export` / ExportedProgram — portable model format | Built with `BUILD_PYTHON=ON` | Deployment-ready model format | None — source preserved | `BUILD_PYTHON=ON` |
| `torch/csrc/inductor/` | DEFERRED_IMPORTANT | AOTInductor / compiled inference path | Built with `BUILD_PYTHON=ON` | Fast compiled CPU inference | None — source preserved | `BUILD_PYTHON=ON` |
| `torch/csrc/dynamo/` | DEFERRED_IMPORTANT | TorchDynamo graph capture and tracing | Built with `BUILD_PYTHON=ON` | JIT compilation, `torch.compile` | None — source preserved | `BUILD_PYTHON=ON` |
| `torch/quantization/` | DEFERRED_IMPORTANT | Python quantization API (`torch.quantization`) | Built with `BUILD_PYTHON=ON` | INT8 model deployment | None — source preserved | `BUILD_PYTHON=ON` |
| `aten/src/ATen/native/quantized/` | DEFERRED_IMPORTANT | Quantization C++ kernels (INT8/FP8 ops) | Built unless `USE_FBGEMM=OFF USE_PYTORCH_QNNPACK=OFF` | Critical for inference optimization | None — source preserved | `USE_FBGEMM=ON` (already ON in CPU build) |
| `aten/src/ATen/cuda/` | OPTIONAL_BACKEND | CUDA tensor ops and kernel dispatch | Compiled when `USE_CUDA=ON` | GPU training/inference | None — source preserved | `USE_CUDA=ON TORCH_CUDA_ARCH_LIST="7.5"` (Colab T4) |
| `aten/src/ATen/hip/` | OPTIONAL_BACKEND | AMD ROCm/HIP ops | Compiled when `USE_ROCM=ON` | AMD GPU | None — source preserved | `USE_ROCM=ON` |
| `aten/src/ATen/mps/` | OPTIONAL_BACKEND | Apple Metal Performance Shaders ops | Compiled when `USE_MPS=ON` (macOS only) | Apple Silicon inference | None — source preserved | `USE_MPS=ON` on macOS |
| `aten/src/ATen/vulkan/` | OPTIONAL_BACKEND | Vulkan GPU ops (mobile/Android) | Compiled when `USE_VULKAN=ON` | Android GPU | None — source preserved | `USE_VULKAN=ON` |
| `aten/src/ATen/xpu/` | OPTIONAL_BACKEND | Intel XPU (Arc GPU) ops | Compiled when `USE_XPU=ON` | Intel GPU | None — source preserved | `USE_XPU=ON` |
| `torch/csrc/distributed/` | OPTIONAL_BACKEND | DDP, FSDP, RPC, c10d collective ops | Compiled when `USE_DISTRIBUTED=ON` | Multi-node training | None — source preserved | `USE_DISTRIBUTED=ON USE_GLOO=ON` |
| `torch/csrc/cuda/` | OPTIONAL_BACKEND | CUDA Python bindings and helpers | Compiled when `USE_CUDA=ON` | GPU Python API | None — source preserved | `USE_CUDA=ON` |
| `torch/csrc/mps/` | OPTIONAL_BACKEND | MPS Python bindings | macOS + `USE_MPS=ON` only | Apple GPU Python API | None — source preserved | `USE_MPS=ON` on macOS |
| `android/` | OPTIONAL_BACKEND | Android NDK build support | Not built on Linux | Mobile deployment | None — source preserved | Android NDK toolchain |
| `functorch/` | OPTIONAL_BACKEND | Functional transforms (vmap, grad) — Python-facing | Built when `BUILD_FUNCTORCH=ON BUILD_PYTHON=ON` | Functional NN research | None — source preserved | `BUILD_FUNCTORCH=ON BUILD_PYTHON=ON` |
| `benchmarks/` | ARCHIVE_ONLY | Performance benchmarks | `BUILD_BENCHMARK=ON` only | Reference perf numbers | None | `BUILD_BENCHMARK=ON` |
| `test/` | ARCHIVE_ONLY | Python and C++ test suite | `BUILD_TEST=ON` only | Regression/correctness testing | None | `BUILD_TEST=ON` |
| `docs/` (pytorch internal) | ARCHIVE_ONLY | PyTorch documentation source (Sphinx) | Never compiled | API reference | None | `cd docs && make html` |
| `tools/packaging/` | ARCHIVE_ONLY | PyPI/conda packaging scripts | Never built | Release packaging | None | Run manually |
| `tools/github/` | ARCHIVE_ONLY | GitHub CI automation scripts | Never built | CI pipeline | None | Run in CI |
| `tools/linter/` | ARCHIVE_ONLY | Linting tools (clang-format, flake8 configs) | Never built | Code quality | None | `lintrunner` |
| `tools/gdb/` | ARCHIVE_ONLY | GDB pretty-printers for tensors | Never built | Debug tooling | None | Source in gdb session |
| `tools/lldb/` | ARCHIVE_ONLY | LLDB pretty-printers | Never built | Debug tooling | None | Source in lldb session |
| `scripts/` | ARCHIVE_ONLY | Miscellaneous build and release scripts | Never built | Reference | None | Run manually |
| `caffe2/perfkernels/` | ARCHIVE_ONLY | Legacy Caffe2 perf kernels (superseded by ATen) | Built only when `USE_FBGEMM=OFF` | Reference implementation | None | Excluded by default when FBGEMM is on |
| `build/` (pytorch internal) | REMOVE_ONLY_IF_SAFE | CMake build output from pytorch's own internal cmake | Generated at build time | None (fully regenerated) | Safe to delete — fully regenerated by `cmake --build` | `cmake --build .` |
| `dist/__pycache__/` | REMOVE_ONLY_IF_SAFE | Python bytecode cache | Auto-generated by Python import | None | Regenerated on next import | Automatic |
| `tools/__pycache__/` | REMOVE_ONLY_IF_SAFE | Python bytecode cache | Auto-generated | None | Regenerated | Automatic |
| `compile_commands.json` | REMOVE_ONLY_IF_SAFE | Compile commands database (clangd index) | Generated by cmake | Only used by IDE (clangd) | Regenerated by cmake | `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` |

---

## ATen native subdirectories

| Path | Class | Notes |
|---|---|---|
| `aten/src/ATen/core/` | CORE_NOW | TensorImpl, Scalar, type dispatch infrastructure |
| `aten/src/ATen/cpu/` | CORE_NOW | CPU vec intrinsics, SIMD dispatch |
| `aten/src/ATen/native/` (CPU .cpp files) | CORE_NOW | All CPU operator implementations |
| `aten/src/ATen/native/quantized/` | DEFERRED_IMPORTANT | Quantized INT8 CPU kernels |
| `aten/src/ATen/native/sparse/` | DEFERRED_IMPORTANT | Sparse tensor ops |
| `aten/src/ATen/native/nested/` | DEFERRED_IMPORTANT | Nested/ragged tensor ops (for transformers) |
| `aten/src/ATen/native/cuda/` | OPTIONAL_BACKEND | CUDA op kernel implementations |
| `aten/src/ATen/native/hip/` | OPTIONAL_BACKEND | ROCm/HIP op kernels |
| `aten/src/ATen/native/mps/` | OPTIONAL_BACKEND | Metal/MPS op kernels |
| `aten/src/ATen/native/xpu/` | OPTIONAL_BACKEND | Intel XPU op kernels |
| `aten/src/ATen/native/vulkan/` | OPTIONAL_BACKEND | Vulkan op kernels |
| `aten/src/ATen/benchmarks/` | ARCHIVE_ONLY | ATen micro-benchmarks |
| `aten/src/ATen/test/` | ARCHIVE_ONLY | ATen C++ unit tests |

---

## Third-party libraries

| Library | Class | Reason | Dependencies | Re-enable |
|---|---|---|---|---|
| `third_party/cpuinfo/` | CORE_NOW | CPU feature detection (SIMD caps, core count) | None | Always on |
| `third_party/fmt/` | CORE_NOW | String formatting used throughout c10 and torch | None | Always on |
| `third_party/pthreadpool/` | CORE_NOW | Thread pool for ATen intra-op parallelism | None | Always on |
| `third_party/concurrentqueue/` | CORE_NOW | Lock-free queue used by c10 | None | Always on |
| `third_party/FP16/` | CORE_NOW | Half-precision float type support | None | Always on |
| `third_party/FXdiv/` | CORE_NOW | Integer division utilities | None | Always on |
| `third_party/psimd/` | CORE_NOW | Portable SIMD abstraction | None | Always on |
| `third_party/sleef/` | CORE_NOW | Vectorized transcendental math (exp, sin, log) | None | Always on |
| `third_party/protobuf/` | CORE_NOW | Serialization for `.pt` model format | None | `BUILD_CUSTOM_PROTOBUF=ON` |
| `third_party/flatbuffers/` | CORE_NOW | FlatBuffers for lite interpreter model format | None | Always on |
| `third_party/nlohmann/` | CORE_NOW | JSON parsing (config, debug output) | None | Always on |
| `third_party/miniz-3.0.2/` | CORE_NOW | ZIP/zlib compression for model archives | None | Always on |
| `third_party/ideep/` | CORE_NOW | oneDNN/MKL-DNN C++ bridge for x86 CPU ops | oneDNN | `USE_MKLDNN=ON` |
| `third_party/XNNPACK/` | CORE_NOW | Optimized mobile/CPU neural network kernels | pthreadpool | `USE_XNNPACK=ON` |
| `third_party/NNPACK/` | CORE_NOW | Neural network compute primitives | None | `USE_NNPACK=ON` |
| `third_party/fbgemm/` | DEFERRED_IMPORTANT | Quantized GEMM — INT8 inference ops | None | `USE_FBGEMM=ON` |
| `third_party/onnx/` | DEFERRED_IMPORTANT | ONNX schema and proto definitions | protobuf | `ONNX_ML=ON USE_SYSTEM_ONNX=OFF` |
| `third_party/pybind11/` | DEFERRED_IMPORTANT | Python/C++ bindings | Python dev headers | `BUILD_PYTHON=ON` |
| `third_party/pocketfft/` | DEFERRED_IMPORTANT | FFT backend for `torch.fft` | None | `USE_FFTW=OFF` (default) |
| `third_party/gloo/` | OPTIONAL_BACKEND | Gloo CPU collective communications | None | `USE_DISTRIBUTED=ON USE_GLOO=ON` |
| `third_party/tensorpipe/` | OPTIONAL_BACKEND | RPC transport layer for distributed | libuv | `USE_TENSORPIPE=ON` |
| `third_party/kineto/` | OPTIONAL_BACKEND | Performance profiling / Chrome trace | libunwind | `USE_KINETO=ON` |
| `third_party/cutlass/` | OPTIONAL_BACKEND | CUDA GEMM/convolution templates | CUDA | `USE_CUDA=ON` |
| `third_party/flash-attention/` | OPTIONAL_BACKEND | Flash attention fused CUDA kernel | CUDA | `USE_CUDA=ON` |
| `third_party/composable_kernel/` | OPTIONAL_BACKEND | ROCm Composable Kernel GEMMs | ROCm | `USE_ROCM_CK_GEMM=ON` |
| `third_party/aiter/` | OPTIONAL_BACKEND | AMD AIter ops | ROCm | `USE_ROCM=ON` |
| `third_party/cudnn_frontend/` | OPTIONAL_BACKEND | cuDNN v8 frontend API | CUDA + cuDNN | `USE_CUDNN=ON` |
| `third_party/NVTX/` | OPTIONAL_BACKEND | NVIDIA profiling markers (nvprof/Nsight) | CUDA | `USE_CUDA=ON` |
| `third_party/VulkanMemoryAllocator/` | OPTIONAL_BACKEND | Vulkan GPU memory allocator | Vulkan SDK | `USE_VULKAN=ON` |
| `third_party/kleidiai/` | OPTIONAL_BACKEND | ARM KleidiAI optimized kernels | AArch64 | `USE_KLEIDIAI=ON` |
| `third_party/mimalloc/` | OPTIONAL_BACKEND | Fast memory allocator (Windows/AArch64) | None | `USE_MIMALLOC=ON` |
| `third_party/llvm-openmp/` | OPTIONAL_BACKEND | LLVM OpenMP runtime (macOS fallback) | None | macOS only |
| `third_party/ittapi/` | OPTIONAL_BACKEND | Intel VTune ITT profiling markers | None | `USE_ITT=ON` (x86 only) |
| `third_party/benchmark/` | ARCHIVE_ONLY | Google Benchmark framework | None | `BUILD_BENCHMARK=ON` |
| `third_party/googletest/` | ARCHIVE_ONLY | Google Test framework | None | `BUILD_TEST=ON` |
| `third_party/cpp-httplib/` | ARCHIVE_ONLY | HTTP client (used by tools only, not libtorch) | None | N/A |
| `third_party/gemmlowp/` | ARCHIVE_ONLY | Legacy quantized GEMM reference (superseded by FBGEMM) | None | N/A |
| `third_party/python-peachpy/` | ARCHIVE_ONLY | x86 assembly codegen tool (build-time only) | Python | N/A |
| `third_party/valgrind-headers/` | ARCHIVE_ONLY | Valgrind client macros for debug builds | None | `USE_VALGRIND=ON` |
| `third_party/mslk/` | ARCHIVE_ONLY | Reference only | None | N/A |

---

## How to re-enable deferred modules

| Module | CMake flags to add | Notes |
|---|---|---|
| ONNX export/import | `ONNX_ML=ON BUILD_PYTHON=ON` | Requires protobuf and pybind11 |
| Quantization (INT8 kernels) | Already included: `USE_FBGEMM=ON` | Python API needs `BUILD_PYTHON=ON` |
| TorchScript JIT | Already compiled into libtorch_cpu | Exposed via C++ `torch::jit` namespace |
| torch.export / ExportedProgram | `BUILD_PYTHON=ON` | Portable deployment format |
| AOTInductor compiled inference | `BUILD_PYTHON=ON` | Requires Python + Inductor |
| torch.compile / TorchDynamo | `BUILD_PYTHON=ON` | Requires Python |
| Distributed (DDP/FSDP) | `USE_DISTRIBUTED=ON USE_GLOO=ON` | CPU collective ops via Gloo |
| CUDA backend (Colab T4) | `USE_CUDA=ON TORCH_CUDA_ARCH_LIST="7.5"` | Requires CUDA 11.x+ toolkit |
| ROCm backend | `USE_ROCM=ON` | Requires ROCm toolkit |
| Apple MPS | `USE_MPS=ON` | macOS 12.3+ only |
| Profiling | `USE_KINETO=ON` | Adds ~50 MB; links libkineto |
| Python bindings | `BUILD_PYTHON=ON` | Requires Python 3.x dev headers |

---

## Invariant

**No source file in `src/core/pytorch/` is deleted by this classification.**
The vendored source is a read-only reference snapshot. All exclusions are enforced via
CMake `option()` flags passed at configure time. The project root build (`ninja -C build dm_core`)
remains unaffected — it uses `find_package(Torch)` against the pre-built `dist/` tree.
