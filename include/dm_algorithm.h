#ifndef DM_ALGORITHM_H
#define DM_ALGORITHM_H

#include "dm_dataset.h"

/**
 * @brief Structure defining an algorithm's interface
 */
typedef struct {
    const char *id;          /**< Unique identifier for the algorithm (e.g., "apriori") */
    const char *name;        /**< Human-readable name */
    const char *description; /**< Brief description of what it does */
    uint32_t supported_types; /**< Bitmask of DM_DatasetType supported by this algorithm */
    
    /**
     * @brief Executes the algorithm
     * @param ds The dataset to process
     * @param params Opaque pointer to algorithm-specific parameters
     * @return DM_Status status code
     */
    DM_Status (*run)(DM_Dataset *ds, void *params);
} DM_Algorithm;

/**
 * @brief Register a new algorithm into the framework
 */
DM_Status dm_register_algorithm(DM_Algorithm *algo);

/**
 * @brief Find an algorithm by its ID
 */
DM_Algorithm* dm_get_algorithm(const char *id);

/**
 * @brief List all registered algorithms
 */
void dm_list_algorithms(void);

#endif // DM_ALGORITHM_H
