#include "core/dm_block.h"
#include "core/dm_dataset.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Loading dataset from file...\n");
    DM_Dataset *ds = dm_dataset_load("dummy_dataset.txt", DM_TYPE_TRANSACTIONAL);
    if (!ds) {
        printf("Failed to load dataset\n");
        return 1;
    }

    printf("Dataset loaded successfully: count=%zu, max_id=%u\n", ds->count, ds->max_id);

    DM_Block block;
    printf("\nConverting Dataset -> DM_Block...\n");
    int rc = dm_dataset_to_block(ds, &block);
    if (rc != 0) {
        printf("Failed to convert dataset to block\n");
        dm_dataset_free(ds);
        return 1;
    }

    printf("DM_Block created successfully!\n");
    printf("Block Details:\n");
    printf(" - Layout: %d (Expected %d)\n", block.layout, DM_LAYOUT_DATASET);
    printf(" - Kind:   %d (Expected %d)\n", block.kind, DM_KIND_EXTERNAL);
    printf(" - Role:   %d (Expected %d)\n", block.role, DM_ROLE_TRANSACTION);
    
    // Print metadata
    for (int i = 0; i < block.meta_count; i++) {
        printf(" - Meta [%s]: %s\n", block.meta[i].key, block.meta[i].value);
    }

    printf("\nUnpacking DM_Block -> Dataset...\n");
    DM_Dataset *unpacked_ds = dm_block_to_dataset(&block);
    if (unpacked_ds) {
        printf("Successfully unpacked! count=%zu, max_id=%u\n", unpacked_ds->count, unpacked_ds->max_id);
        if (unpacked_ds == ds) {
            printf("Integrity verified: Pointer perfectly matches original dataset.\n");
        } else {
            printf("Integrity failed: Pointers mismatch.\n");
        }
    } else {
        printf("Failed to unpack dataset from block\n");
    }

    // Clean up
    dm_block_free(&block); // Should safely do nothing to the payload since owns_data = 0
    dm_dataset_free(ds);   // Actually free the dataset

    printf("\nClean up complete.\n");
    return 0;
}
