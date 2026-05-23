# Migration of `dm_engine` and Models to `DM_Block`

This plan has been refined to follow a **4-Phase safe migration strategy**. Instead of deleting `DM_Tensor` immediately and causing widespread compilation errors, we will introduce a temporary shim, migrate components incrementally, and only delete the legacy API when all tests pass.

## Phase 0: Foundations & Shims (Done)
- **DM_Block Enhancements**:
  - Add `int owns_handle;` to `DM_Block` to explicitly control handle lifecycle.
  - Add shape helper functions to `dm_block.h`: `dm_block_dim()`, `dm_block_is_nchw4()`.
  - Add macros for easy dimension access: `DM_NCHW_N`, `DM_NCHW_C`, etc.
- **Lowering Policy**:
  - Define `DM_LowerPolicy` (`COPY`, `VIEW`, `MOVE`) in `dm_lowering.h` and update `dm_lower_block` signatures.
- **Deprecation Shim**:
  - Mark `DM_Tensor` related functions with a deprecation notice. We will retain the `DM_Tensor` struct temporarily so legacy code compiles while we migrate it.

## Phase 1: Refactor Engine Primitives (Done)
- Update `dm_engine.h` and `dm_engine.c`.
- Change `dm_op_conv2d_same`, `dm_op_matmul_nt`, etc., to accept `DM_Block *`.
- Inside `dm_engine.c`, use `dm_lower_block(..., DM_LOWER_VIEW)` to fetch TensorFlow handles.
- Add strict validation (layout, dtype, kind) inside primitive ops.

## Phase 2: Refactor Models
- Update signatures in `src/models/` (e.g., `dm_resnet18_forward`, `dm_tinyvit_forward`) to accept `DM_Block *`.
- Replace all legacy `->n`, `->c`, `->h`, `->w` accesses with the new shape macros/helpers.
- Replace internal tensor allocations with `dm_block_create`.

## Phase 3: Update Ecosystem
- Update CLI, benchmarks, and examples (e.g., `resnet18 bench`) to use `DM_Block` from end to end.
- Verify everything compiles cleanly and runs correctly.

## Phase 4: Eradicate Legacy API
- Delete `DM_Tensor` from `include/dm.h` and `include/core/dm_engine.h`.
- Delete `dm_tensor_alloc`, `dm_tensor_free`, `dm_tensor_set`.
- Update README to note the breaking v0.x change.
