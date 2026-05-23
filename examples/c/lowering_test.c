#include "core/dm_block.h"
#include "lowering/dm_lowering.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    DM_Block b;
    int64_t shape[] = {2, 3};
    if (dm_block_create(&b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, shape) == 0) {
        printf("Created CPU block: count=%zu, bytes=%zu\n", b.count, b.bytes);
        
        // Populate with some dummy data
        float *data = (float*)b.data;
        for (int i = 0; i < 6; ++i) {
            data[i] = (float)i;
        }

        DM_Block tf_block;
        int rc = dm_lower_block(&b, &tf_block, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW);
        if (rc == 0) {
            printf("Successfully lowered to TensorFlow block. Backend: %d, Layout: %d, Kind: %d\n", 
                   tf_block.backend, tf_block.layout, tf_block.kind);
                   
            DM_Block raised_block;
            rc = dm_raise_block(&tf_block, &raised_block, DM_BACKEND_CPU, DM_LOWER_VIEW);
            if (rc == 0) {
                printf("Successfully raised from TensorFlow to CPU block.\n");
                float *raised_data = (float*)raised_block.data;
                int match = 1;
                for (int i = 0; i < 6; ++i) {
                    if (raised_data[i] != data[i]) match = 0;
                }
                if (match) {
                    printf("Data integrity verified! Data successfully round-tripped through TF C API.\n");
                } else {
                    printf("Data mismatch!\n");
                }
                dm_block_free(&raised_block);
            } else {
                printf("Failed to raise from TensorFlow\n");
            }
        } else {
            printf("Failed to lower to TensorFlow\n");
        }
        dm_block_free(&b);
    } else {
        printf("Failed to create CPU block\n");
    }
    return 0;
}
