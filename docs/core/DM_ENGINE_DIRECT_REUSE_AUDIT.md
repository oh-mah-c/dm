# DM Engine Direct Reuse Audit

Status: Phase 0 completed with a hard build prerequisite identified.

## Goal

Use the Tensor/TensorFlow core already present under:

```text
src/core/dm_engine/
  tensorflow/
  third_party/
```

The desired direction is Tensor-first across C/C++, tokenizer, data processing,
models, ops, and training. This audit checks whether the checked-in
`dm_engine` tree can be used directly without first writing a new runtime.

## Immediate Result

A direct C++ smoke test that includes TensorFlow core Tensor headers does not
compile yet.

Smoke source:

```text
scratch/tf_tensor_smoke.cc
```

Command attempted:

```sh
g++ -std=c++17 \
  -Isrc/core/dm_engine \
  -Isrc/core/dm_engine/third_party/xla \
  -Isrc/core/dm_engine/third_party/xla/third_party/tsl \
  -c scratch/tf_tensor_smoke.cc \
  -o build/obj/tf_tensor_smoke.o
```

Observed failure:

```text
src/core/dm_engine/tensorflow/core/framework/tensor.h:26:10:
fatal error: unsupported/Eigen/CXX11/Tensor: No such file or directory
```

Additional checked prerequisites are also missing:

```text
src/core/dm_engine/tensorflow/core/framework/types.pb.h        missing
src/core/dm_engine/tensorflow/core/framework/tensor_shape.pb.h missing
src/core/dm_engine/tensorflow/core/framework/tensor.pb.h       missing
src/core/dm_engine/tensorflow/core/framework/full_type.pb.h    missing
```

So the current `dm_engine` checkout is not yet a self-contained buildable
TensorFlow core tree. It contains TensorFlow source files, but not all generated
headers and not Eigen.

## Tensor Files Present

Important Tensor/framework sources exist under:

```text
src/core/dm_engine/tensorflow/core/framework/tensor.h
src/core/dm_engine/tensorflow/core/framework/tensor.cc
src/core/dm_engine/tensorflow/core/framework/tensor_shape.h
src/core/dm_engine/tensorflow/core/framework/tensor_shape.cc
src/core/dm_engine/tensorflow/core/framework/tensor_types.h
src/core/dm_engine/tensorflow/core/framework/types.h
src/core/dm_engine/tensorflow/core/framework/types.cc
src/core/dm_engine/tensorflow/core/framework/allocator.h
src/core/dm_engine/tensorflow/core/framework/allocator_registry.h
src/core/dm_engine/tensorflow/core/framework/typed_allocator.h
src/core/dm_engine/tensorflow/core/framework/typed_allocator.cc
```

These are the right Tensor core files, but they depend on generated protobuf
headers and TensorFlow's normal third-party dependency setup.

## Required Runtime Support

The Tensor core headers pull in at least:

```text
unsupported/Eigen/CXX11/Tensor
tensorflow/core/framework/*.pb.h
tensorflow/core/lib/core/refcount.h
tensorflow/core/lib/core/status.h
tensorflow/core/lib/core/stringpiece.h
tensorflow/core/lib/gtl/inlined_vector.h
tensorflow/core/platform/mem.h
tensorflow/core/platform/types.h
tensorflow/core/platform/status.h
tensorflow/core/platform/statusor.h
tensorflow/core/platform/logging.h
tensorflow/core/platform/errors.h
absl/*
xla/tsl/*
```

Current blockers:

- Eigen include tree is not present in the searched paths.
- TensorFlow generated protobuf headers are not present.
- `protoc` is not available on PATH.
- Python `tensorflow` is not installed in this environment, despite the current
  Makefile referencing `.venv/lib/python3.12/site-packages/tensorflow`.
- The repo build is Makefile/gcc-first; TensorFlow core is C++ and normally
  expects Bazel-generated include products.

## First Compilable Tensor Subset

The first realistic subset is still:

```text
tensorflow/core/framework/tensor*
tensorflow/core/framework/tensor_shape*
tensorflow/core/framework/types*
tensorflow/core/framework/allocator*
tensorflow/core/platform/*
tensorflow/core/lib/core/*
tensorflow/core/lib/gtl/*
xla/tsl/*
absl/*
Eigen unsupported Tensor headers
generated *.pb.h / *.pb.cc from TensorFlow proto files
```

But this subset cannot compile until Eigen and generated protobuf headers are
available.

## Ops Found For Reuse

Potential TensorFlow CPU op sources exist under:

```text
src/core/dm_engine/tensorflow/core/kernels/cwise_op_add_*.cc
src/core/dm_engine/tensorflow/core/kernels/matmul_op_real.cc
src/core/dm_engine/tensorflow/core/kernels/matmul_op_impl.h
src/core/dm_engine/tensorflow/core/kernels/relu_op.cc
src/core/dm_engine/tensorflow/core/kernels/softmax_op.cc
src/core/dm_engine/tensorflow/core/kernels/conv_ops.cc
src/core/dm_engine/tensorflow/core/kernels/conv_ops_float.cc
src/core/dm_engine/tensorflow/core/kernels/maxpooling_op.cc
src/core/dm_engine/tensorflow/core/kernels/avgpooling_op.cc
src/core/dm_engine/tensorflow/core/kernels/bias_op.cc
src/core/dm_engine/tensorflow/core/kernels/xent_op.cc
src/core/dm_engine/tensorflow/core/kernels/sparse_xent_op.cc
```

Image/data preprocessing candidates:

```text
src/core/dm_engine/tensorflow/core/kernels/image/resize_nearest_neighbor_op.cc
src/core/dm_engine/tensorflow/core/kernels/image/resize_bilinear_op.cc
src/core/dm_engine/tensorflow/core/kernels/image/decode_image_op.cc
src/core/dm_engine/tensorflow/core/kernels/image/colorspace_op.cc
src/core/dm_engine/tensorflow/core/kernels/image/crop_and_resize_op.cc
src/core/dm_engine/tensorflow/core/kernels/data/tensor_dataset_op.cc
src/core/dm_engine/tensorflow/core/kernels/data/tensor_slice_dataset_op.cc
src/core/dm_engine/tensorflow/core/kernels/data/batch_dataset_op style files
```

Risk:

Most TensorFlow op kernels depend on `OpKernel`, op registration, device
context, allocators, Eigen threadpool, generated op definitions, and protobuf
metadata. They are not simple standalone functions.

## Training And Gradients

Training-related files exist:

```text
src/core/dm_engine/tensorflow/core/kernels/training_ops.cc
src/core/dm_engine/tensorflow/core/kernels/training_ops.h
src/core/dm_engine/tensorflow/core/kernels/variable_ops.cc
src/core/dm_engine/tensorflow/core/kernels/resource_variable_ops.cc
src/core/dm_engine/tensorflow/cc/framework/gradients.cc
src/core/dm_engine/tensorflow/cc/gradients/math_grad.cc
src/core/dm_engine/tensorflow/cc/gradients/nn_grad.cc
src/core/dm_engine/tensorflow/cc/gradients/image_grad.cc
```

Direct reuse is likely expensive because this area normally requires:

- graph/runtime objects
- op registry
- function library
- resource manager
- variables/resources
- generated op/proto files
- TensorFlow C++ runtime build configuration

Training should not be attempted before the base Tensor compile works.

## Tokenizer/Text Tensor Plan

Existing tokenizer surfaces include:

```text
include/tokenizer/tokenizer.h
include/tokenizer/dm_tokenizer_block.h
src/tokenizer/*
src/models/tinystories.c
```

Tensor-first representation should be:

```text
input_ids:      int32/int64 Tensor [batch, seq]
attention_mask: int32/int64 Tensor [batch, seq]
labels:         int32/int64 Tensor [batch, seq] or [batch]
```

The existing `dm_tokenizer_block.h` path should be replaced by Tensor output
once the Tensor import surface is buildable.

## Model Migration Targets

Model files currently tied to older project containers include:

```text
src/models/gan.c
src/models/mlp_train.c
src/models/tinystories.c
src/models/vision/vit.c
src/models/vision/swin.c
src/models/vision/resnet.c
src/models/vision/mobilenet_tiny.c
```

The migration target is:

- Tensor inputs
- Tensor weights/parameters
- Tensor intermediates
- Tensor outputs
- TensorFlow core ops where they can be compiled/imported directly

Do not start this migration until Phase 1 proves Tensor core compilation.

## Current Public API Conflict

Current public headers still expose older project containers in AI ops:

```text
include/dm.h
include/core/dm_engine.h
include/core/dm_block.h
include/tokenizer/dm_tokenizer_block.h
include/core/dm_dataset.h
```

These files must become Tensor-first later, but changing them before Tensor core
builds would only replace one broken surface with another.

## What Can Be Built Now

The existing Makefile is C/gcc-first and links TensorFlow shared libraries from:

```text
.venv/lib/python3.12/site-packages/tensorflow
```

That is different from compiling TensorFlow core from `src/core/dm_engine`.

Direct TensorFlow core compilation from `src/core/dm_engine` currently cannot
proceed because Eigen and generated protobuf headers are absent.

## Smallest Safe Next Step

Before Phase 1 can be completed, add or generate the missing TensorFlow build
inputs:

1. Add Eigen include tree containing `unsupported/Eigen/CXX11/Tensor`.
2. Generate or import TensorFlow protobuf outputs such as:
   - `types.pb.h`
   - `tensor_shape.pb.h`
   - `tensor.pb.h`
   - `full_type.pb.h`
3. Re-run the smoke compile:

```sh
g++ -std=c++17 \
  -Isrc/core/dm_engine \
  -Isrc/core/dm_engine/third_party/xla \
  -Isrc/core/dm_engine/third_party/xla/third_party/tsl \
  -c scratch/tf_tensor_smoke.cc \
  -o build/obj/tf_tensor_smoke.o
```

Only after this succeeds should the Makefile gain a permanent Tensor core smoke
target.
