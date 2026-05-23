#include "core/dm_block.h"
#include <stdio.h>

int main() {
    DM_Block b;
    int64_t shape[] = {2, 3};
    if (dm_block_create(&b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, shape) == 0) {
        printf("Created block: count=%zu, bytes=%zu\n", b.count, b.bytes);
        dm_block_set_meta(&b, "key", "value");
        printf("Meta: %s = %s\n", b.meta[0].key, b.meta[0].value);
        dm_block_free(&b);
        printf("Freed block\n");
    } else {
        printf("Failed to create block\n");
    }
    return 0;
}
