#include "core/dm_algorithm.h"
#include <string.h>

#define MAX_ALGORITHMS 128

static DM_Algorithm *registry[MAX_ALGORITHMS];
static int algorithm_count = 0;

DM_Status dm_register_algorithm(DM_Algorithm *algo) {
    if (!algo || !algo->id) return DM_ERROR_INVALID_PARAM;
    if (algorithm_count >= MAX_ALGORITHMS) return DM_ERROR_MEMORY;
    
    // Check for duplicates
    for (int i = 0; i < algorithm_count; i++) {
        if (strcmp(registry[i]->id, algo->id) == 0) {
            return DM_SUCCESS; // Already registered
        }
    }
    
    registry[algorithm_count++] = algo;
    return DM_SUCCESS;
}

DM_Algorithm* dm_get_algorithm(const char *id) {
    if (!id) return NULL;
    for (int i = 0; i < algorithm_count; i++) {
        if (strcmp(registry[i]->id, id) == 0) {
            return registry[i];
        }
    }
    return NULL;
}

void dm_list_algorithms(void) {
    printf("\nAvailable Algorithms:\n");
    printf("====================\n");
    for (int i = 0; i < algorithm_count; i++) {
        printf("[%s] %s\n", registry[i]->id, registry[i]->name);
        printf("    Desc: %s\n", registry[i]->description);
    }
    printf("====================\n\n");
}
