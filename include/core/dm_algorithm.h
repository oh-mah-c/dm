#ifndef DM_ALGORITHM_H
#define DM_ALGORITHM_H

#include "dm_dataset.h"

typedef struct {
    const char *id;
    const char *name;
    const char *description;
    uint32_t supported_types; 
    
    DM_Status (*run)(DM_Dataset *ds, void *params);
} DM_Algorithm;

/**
 * @brief Registry API
 */
DM_Status dm_register_algorithm(DM_Algorithm *algo);
DM_Algorithm* dm_get_algorithm(const char *id);
void dm_list_algorithms(void);

/**
 * @brief Helper macro for automatic registration
 * Use this in your algorithm .c files.
 */
#if defined(__GNUC__) || defined(__clang__)
#define DM_REGISTER_ALGORITHM(algo_var) \
    __attribute__((constructor)) static void _reg_##algo_var() { \
        dm_register_algorithm(&algo_var); \
    }
#else
#define DM_REGISTER_ALGORITHM(algo_var) \
    /* Manual registration required for this compiler */
#endif

#endif // DM_ALGORITHM_H
