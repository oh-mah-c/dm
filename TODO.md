# TODO: Direct Tensor Core From `src/core/dm_engine`

## Direction

Use the Tensor/TensorFlow core already present inside:

```text
src/core/dm_engine/
  tensorflow/
  third_party/
```

Do not rewrite TensorFlow ops into new DM ops.
Do not make a large mapping layer around Tensor.

The goal now is:

- use Tensor as the main runtime object
- expose/import the existing Tensor core from `dm`
- focus first on C/C++
- make tokenizer, data processing, models, ops, and training operate on Tensor
- keep code movement minimal
- only add the smallest build/API glue needed to compile and use the existing core

Tensor is the single runtime data object.

---

## Non-Goals

- No pure-C reimplementation of TensorFlow kernels.
- No large wrapper system that recreates TensorFlow APIs.
- No new model/layer abstraction before Tensor import works.
- No VGG/Swin/model migration before Tensor core is usable.
- No rewriting TensorFlow internals unless required to make the selected core build.

---

## Phase 0: Audit `dm_engine` For Direct Reuse

Status: completed

Purpose:

Find the smallest subset of `src/core/dm_engine/` that can be built/imported for Tensor, ops, data, gradients, and training support.

Tasks:

- [x] Map TensorFlow tensor files:
  - `tensorflow/core/framework/tensor*`
  - `tensorflow/core/framework/types*`
  - `tensorflow/core/framework/tensor_shape*`
  - `tensorflow/core/framework/allocator*`
- [x] Map required runtime support:
  - status/error handling
  - env/platform layer
  - memory allocators
  - logging/check macros
  - threadpool/eigen dependencies
- [x] Map math/op kernel files needed first:
  - matmul
  - conv2d
  - relu/activation
  - softmax
  - reductions
  - image preprocessing
- [x] Map training-related files:
  - gradient registry
  - optimizer-related code
  - loss functions
  - variables/resources
- [x] Map tokenizer/text files and decide the Tensor representation for token ids, masks, and sequence batches.
- [x] Map model entry points that must become Tensor-first:
  - MLP
  - GAN
  - TinyStories
  - image models
  - training examples
- [x] Identify what cannot be directly reused without Bazel/protobuf/codegen.
- [x] Identify what can be built with the current project build system.
- [x] Write the result to `docs/core/DM_ENGINE_DIRECT_REUSE_AUDIT.md`.

Acceptance:

- [x] We know the first compilable Tensor subset.
- [x] We know which dependencies are required.
- [x] We know which files are too expensive to pull in now.
- [x] No runtime architecture rewrite has started beyond choosing Tensor as the single object.

---

## Phase 1: Build The Existing Tensor Core First

Status: blocked: missing Eigen include tree, generated TensorFlow protobuf headers, and `protoc`

Purpose:

Make the actual Tensor core from `src/core/dm_engine/tensorflow/core/framework` compile inside the `dm` project.

Tasks:

- [ ] Add a dedicated build target for the selected TensorFlow core subset.
- [ ] Include only required Tensor core/framework/platform files.
- [ ] Include required `third_party` dependencies already present under `dm_engine`.
- [ ] Avoid copying files again unless necessary.
- [ ] Prefer building files from their current location:

```text
src/core/dm_engine/tensorflow/...
src/core/dm_engine/third_party/...
```

- [x] Produce a small C++ smoke test that creates a Tensor directly from TensorFlow core.
- [ ] Do not expose a DM wrapper API yet.

Acceptance:

- [ ] A C++ test can include TensorFlow core headers from `dm_engine`.
- [ ] A Tensor can be allocated and inspected.
- [ ] The build proves the imported core works before higher-level APIs are touched.

---

## Phase 2: Expose Tensor Core As A DM Import Surface

Status: pending

Purpose:

Make users able to import Tensor core from `dm` without learning the internal folder layout.

Important:

This is not a runtime wrapper.
This is only an import/build surface.

Tasks:

- [ ] Add a public C++ import header, for example:

```text
include/dm/tensor_core.hpp
```

- [ ] That header should directly expose the selected TensorFlow Tensor types/namespaces.
- [ ] Keep it thin: includes, aliases, and build configuration only.
- [ ] Avoid recreating Tensor behavior in DM structs.
- [ ] Document how to include and link it.

Example target direction:

```cpp
#include <dm/tensor_core.hpp>

tensorflow::Tensor x;
```

Acceptance:

- [ ] External C++ code can import Tensor from `dm`.
- [ ] No duplicate DM Tensor implementation is required for the first milestone.

---

## Phase 3: Decide The C Boundary

Status: pending

Purpose:

TensorFlow core Tensor is C++.
C cannot directly own/use C++ classes without some ABI boundary.

The C/C++ focus should be:

1. C++ first: direct TensorFlow core import.
2. C second: minimal opaque-handle ABI only if needed.

Tasks:

- [ ] Decide whether plain C must use Tensor immediately.
- [ ] If C is required, expose only opaque handles:

```c
typedef struct DM_TensorHandle DM_TensorHandle;
```

- [ ] Keep the C handle minimal:
  - create tensor
  - destroy tensor
  - get shape
  - get data pointer
  - maybe run selected ops
- [ ] Do not recreate all TensorFlow Tensor APIs in C.

Acceptance:

- [ ] C++ path works directly.
- [ ] C path, if added, is small and opaque.
- [ ] No large wrapper/runtime is created.

---

## Phase 4: Bring Up Existing Ops From `dm_engine`

Status: pending

Purpose:

Use TensorFlow's existing op/kernel code where practical instead of rewriting ops.

Tasks:

- [ ] Start with CPU float32 ops only.
- [ ] Bring up one op at a time:
  - identity/copy
  - add
  - matmul
  - relu
  - conv2d
  - softmax
- [ ] For each op, identify required kernel/runtime dependencies.
- [ ] If an op drags in too much runtime, document it and postpone.
- [ ] Do not replace it with a new DM reimplementation unless explicitly approved.

Acceptance:

- [ ] At least one TensorFlow core op can run through the imported core.
- [ ] Dependency cost is known per op.
- [ ] Build remains understandable.

---

## Phase 5: Data Processing From `dm_engine`

Status: pending

Purpose:

Use existing TensorFlow data/image preprocessing code where feasible.

Tasks:

- [ ] Audit usable image/data files under:

```text
tensorflow/core/kernels/image/
tensorflow/core/lib/
tensorflow/core/platform/
```

- [ ] Bring up only the pieces that can compile without the full TensorFlow runtime.
- [ ] Focus first on:
  - image decode/load path if practical
  - resize
  - normalize/cast
  - batching helpers if available
- [ ] Avoid recreating a `tf.data`-style system.

Acceptance:

- [ ] A Tensor-backed data preprocessing path is available or documented as too expensive.
- [ ] Data outputs are Tensor objects directly.

---

## Phase 6: Tokenizer And Text Pipeline To Tensor

Status: pending

Purpose:

Make tokenizer/text processing produce Tensor objects directly, so language models and TinyStories do not use a separate project-specific runtime object.

Tasks:

- [ ] Audit existing tokenizer code and public tokenizer APIs.
- [ ] Decide Tensor dtypes for text:
  - token ids
  - attention masks
  - segment/type ids if needed
  - labels/targets
- [ ] Make tokenizer batch output shape explicit:

```text
input_ids:      [batch, seq]
attention_mask: [batch, seq]
labels:         [batch, seq] or [batch]
```

- [ ] Prefer TensorFlow core Tensor storage/import where practical.
- [ ] Avoid adding a new tokenizer wrapper object that duplicates Tensor.
- [ ] Add a C++ smoke test that tokenizes text into Tensor-backed arrays.

Acceptance:

- [ ] Tokenizer output can be consumed by Tensor-first model code.
- [ ] Text batching has explicit Tensor shapes and dtypes.
- [ ] No separate runtime container is introduced for tokens.

---

## Phase 7: Models To Tensor

Status: pending

Purpose:

Move project models to Tensor-first inputs, parameters, intermediates, and outputs.

Tasks:

- [ ] Audit model files:
  - `src/models/gan.c`
  - `src/models/mlp_train.c`
  - `src/models/tinystories.c`
  - vision/model examples
- [ ] For every model, define Tensor inputs and Tensor outputs first.
- [ ] Store weights/parameters as Tensor objects.
- [ ] Use imported TensorFlow core ops when feasible.
- [ ] If an op cannot be imported yet, pause the model phase until the op phase resolves it.
- [ ] Do not introduce a separate layer/module object as the primary execution API.

Acceptance:

- [ ] Tiny MLP forward path is Tensor-first.
- [ ] TinyStories data/model path receives Tensor token batches.
- [ ] GAN tensors are explicit for noise, generated samples, discriminator inputs, and losses.
- [ ] Model code does not depend on a separate project runtime container.

---

## Phase 8: Training/Gradient Feasibility

Status: pending

Purpose:

Check whether TensorFlow training/gradient code can be reused directly without importing too much of TensorFlow runtime.

Tasks:

- [ ] Audit gradient and optimizer dependencies.
- [ ] Identify whether gradient execution requires:
  - graph runtime
  - eager runtime
  - op registry
  - resource manager
  - function library
  - protobuf graph machinery
- [ ] If direct reuse is practical, add the minimal build subset.
- [ ] If direct reuse requires too much runtime, document that clearly before choosing another path.

Acceptance:

- [ ] We know whether training can be directly reused.
- [ ] No manual training rewrite starts before this decision.

---

## Phase 9: Legacy API Cleanup

Status: pending

Purpose:

Keep old project code compiling while the Tensor core import becomes the main path, then remove abandoned APIs after review.

Tasks:

- [ ] Stop adding APIs that use a non-Tensor runtime object.
- [ ] Keep old examples/tests only as temporary compatibility.
- [ ] Prefer new C++ examples using direct Tensor import.
- [ ] Remove or quarantine abandoned pure-C reimplementation work after review.
- [ ] Make tokenizer, data, models, ops, and training docs use Tensor terminology only.

Acceptance:

- [ ] New code uses Tensor core directly.
- [ ] Old code still compiles unless explicitly removed.
- [ ] The project direction is no longer split between Tensor-first and older project-specific containers.

---

## First Concrete Task

Start with Phase 0 only.

Create:

```text
docs/core/DM_ENGINE_DIRECT_REUSE_AUDIT.md
```

The audit must answer:

1. Which TensorFlow Tensor files can be used directly from `src/core/dm_engine/`?
2. What minimum platform/runtime files are required?
3. What minimum third-party files are required?
4. Can a C++ program create and inspect `tensorflow::Tensor` from this repo?
5. Which ops can be compiled with low dependency cost?
6. Which ops require the full TensorFlow runtime?
7. Is training/gradient reuse practical directly?
8. What public include path should `dm` expose first?
9. Which tokenizer/model/data APIs must become Tensor-first?
10. What build target proves this path works?

After Phase 0, stop and review before changing build code.
