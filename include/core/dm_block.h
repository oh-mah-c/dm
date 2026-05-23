#ifndef DM_BLOCK_H
#define DM_BLOCK_H

#include "dm_dtype.h"
#include "dm_layout.h"
#include "dm_backend.h"
#include "dm_role.h"
#include "dm_meta.h"
#include "dm_invariant.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DM_MAX_DIMS 8
#define DM_MAX_META 32
#define DM_MAX_NAME 128
#define DM_MAX_INVARIANTS 16

typedef struct {
    DM_BlockKind kind;
    DM_DType dtype;
    DM_Layout layout;
    DM_Backend backend;
    DM_Role role;

    void *data;
    void *handle;

    int ndim;
    int64_t shape[DM_MAX_DIMS];
    int64_t stride[DM_MAX_DIMS];

    size_t count;
    size_t bytes;

    int owns_data;
    int owns_handle;
    void (*handle_destructor)(void *handle);
    int requires_grad;

    void *grad;
    void *aux;

    DM_MetaItem meta[DM_MAX_META];
    int meta_count;

    DM_Invariant invariants[DM_MAX_INVARIANTS];
    int invariant_count;

    char name[DM_MAX_NAME];
} DM_Block;

int dm_block_create(
    DM_Block *block,
    DM_BlockKind kind,
    DM_DType dtype,
    DM_Layout layout,
    DM_Backend backend,
    int ndim,
    const int64_t *shape
);

int dm_block_view(
    DM_Block *block,
    DM_BlockKind kind,
    DM_DType dtype,
    DM_Layout layout,
    DM_Backend backend,
    int ndim,
    const int64_t *shape,
    void *data
);

void dm_block_free(DM_Block *block);

int dm_block_to_backend(
    const DM_Block *src,
    DM_Block *dst,
    DM_Backend backend
);

int dm_block_set_meta(
    DM_Block *block,
    const char *key,
    const char *value
);

/* ── Shape Helpers ──────────────────────────────────────────────────────── */
int64_t dm_block_dim(const DM_Block *b, int axis);
int dm_block_is_nchw4(const DM_Block *b);
int dm_block_is_nhwc4(const DM_Block *b);

/* Quick NCHW macros */
#define DM_NCHW_N(b) ((b)->shape[0])
#define DM_NCHW_C(b) ((b)->shape[1])
#define DM_NCHW_H(b) ((b)->shape[2])
#define DM_NCHW_W(b) ((b)->shape[3])

#ifdef __cplusplus
}
#endif

#endif // DM_BLOCK_H
