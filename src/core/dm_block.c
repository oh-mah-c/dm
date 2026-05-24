#include "dm_block.h"
#include <stdlib.h>
#include <string.h>

int dm_block_create(
    DM_Block *block,
    DM_BlockKind kind,
    DM_DType dtype,
    DM_Layout layout,
    DM_Backend backend,
    int ndim,
    const int64_t *shape)
{
    if (!block) return -1;
    memset(block, 0, sizeof(DM_Block));
    
    block->kind = kind;
    block->dtype = dtype;
    block->layout = layout;
    block->backend = backend;
    block->role = DM_ROLE_UNKNOWN;
    
    if (ndim < 0 || ndim > DM_MAX_DIMS) return -1;
    block->ndim = ndim;
    
    size_t total_count = 1;
    for (int i = 0; i < ndim; ++i) {
        block->shape[i] = shape[i];
        total_count *= shape[i];
    }
    block->count = ndim == 0 ? 0 : total_count;
    
    int64_t current_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        block->stride[i] = current_stride;
        current_stride *= block->shape[i];
    }
    
    block->bytes = block->count * dm_dtype_size(dtype);
    
    if (backend == DM_BACKEND_CPU && block->bytes > 0 && layout != DM_LAYOUT_NONE && layout != DM_LAYOUT_TF_HANDLE) {
        block->data = calloc(1, block->bytes);
        if (!block->data) return -1;
        block->owns_data = 1;
    }
    block->dirty = 1;
    block->version = 1;
    return 0;
}

int dm_block_view(
    DM_Block *block,
    DM_BlockKind kind,
    DM_DType dtype,
    DM_Layout layout,
    DM_Backend backend,
    int ndim,
    const int64_t *shape,
    void *data)
{
    if (!block) return -1;
    memset(block, 0, sizeof(DM_Block));
    
    block->kind = kind;
    block->dtype = dtype;
    block->layout = layout;
    block->backend = backend;
    block->role = DM_ROLE_UNKNOWN;
    
    if (ndim < 0 || ndim > DM_MAX_DIMS) return -1;
    block->ndim = ndim;
    
    size_t total_count = 1;
    for (int i = 0; i < ndim; ++i) {
        block->shape[i] = shape[i];
        total_count *= shape[i];
    }
    block->count = ndim == 0 ? 0 : total_count;
    
    int64_t current_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        block->stride[i] = current_stride;
        current_stride *= block->shape[i];
    }
    
    block->bytes = block->count * dm_dtype_size(dtype);
    block->data = data;
    block->owns_data = 0;
    block->owns_handle = 0;
    block->dirty = 1;
    block->version = 1;
    
    return 0;
}

void dm_block_free(DM_Block *block) {
    if (!block) return;
    
    // 1. Free handle if owned and destructor provided
    if (block->owns_handle && block->handle && block->handle_destructor) {
        block->handle_destructor(block->handle);
        block->handle = NULL;
    }
    block->owns_handle = 0;

    // 2. Free data if owned
    if (block->owns_data && block->data) {
        free(block->data);
        block->data = NULL;
    }
    block->owns_data = 0;
}

int dm_block_set_meta(DM_Block *block, const char *key, const char *value) {
    if (!block || !key || !value) return -1;
    
    for (int i = 0; i < block->meta_count; ++i) {
        if (strcmp(block->meta[i].key, key) == 0) {
            strncpy(block->meta[i].value, value, sizeof(block->meta[i].value) - 1);
            block->meta[i].value[sizeof(block->meta[i].value) - 1] = '\0';
            return 0;
        }
    }
    
    if (block->meta_count >= DM_MAX_META) return -1;
    
    strncpy(block->meta[block->meta_count].key, key, sizeof(block->meta[block->meta_count].key) - 1);
    block->meta[block->meta_count].key[sizeof(block->meta[block->meta_count].key) - 1] = '\0';
    
    strncpy(block->meta[block->meta_count].value, value, sizeof(block->meta[block->meta_count].value) - 1);
    block->meta[block->meta_count].value[sizeof(block->meta[block->meta_count].value) - 1] = '\0';
    
    block->meta_count++;
    return 0;
}

int dm_block_to_backend(const DM_Block *src, DM_Block *dst, DM_Backend backend) {
    if (!src || !dst) return -1;
    if (src->backend == backend) {
        *dst = *src;
        dst->owns_data = 0;
        return 0;
    }
    return -1;
}

int64_t dm_block_dim(const DM_Block *b, int axis) {
    if (!b || axis < 0 || axis >= b->ndim) return 0;
    return b->shape[axis];
}

int dm_block_is_nchw4(const DM_Block *b) {
    if (!b || b->ndim != 4) return 0;
    return (b->layout == DM_LAYOUT_NCHW || b->layout == DM_LAYOUT_ROW_MAJOR);
}

int dm_block_is_nhwc4(const DM_Block *b) {
    if (!b || b->ndim != 4) return 0;
    return b->layout == DM_LAYOUT_NHWC;
}
