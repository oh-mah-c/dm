#include "lowering/dm_lower_cpu.h"
#include <string.h>
#include <stdlib.h>

int dm_block_to_cpu(const DM_Block *src, DM_Block *dst) {
    if (!src || !dst) return -1;
    if (src->backend == DM_BACKEND_CPU) {
        *dst = *src;
        dst->owns_data = 0; // Shallow copy
        return 0;
    }
    
    // Copy structurally and allocate memory
    *dst = *src;
    dst->backend = DM_BACKEND_CPU;
    
    if (src->bytes > 0 && src->data) {
        dst->data = malloc(src->bytes);
        if (!dst->data) return -1;
        memcpy(dst->data, src->data, src->bytes);
        dst->owns_data = 1;
    } else {
        dst->data = NULL;
        dst->owns_data = 0;
    }
    
    return 0;
}

int dm_cpu_to_block(const DM_Block *src, DM_Block *dst) {
    if (!src || !dst) return -1;
    *dst = *src;
    dst->owns_data = 0; // Shallow copy
    return 0;
}
