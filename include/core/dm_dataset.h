#ifndef DM_DATASET_H
#define DM_DATASET_H

#include "dm_common.h"
#include "dm_dataset_types.h"

typedef enum {
    DM_TYPE_TRANSACTIONAL, 
    DM_TYPE_UTILITY,       
    DM_TYPE_MATRIX,
    DM_TYPE_SEQUENCE_UTILITY
} DM_DatasetType;

/**
 * @brief Polymorphic dataset container
 */
typedef struct {
    DM_DatasetType type;
    size_t count;        /**< Number of records (rows, transactions, etc.) */
    uint32_t max_id;     /**< Maximum item ID found (if applicable) */
    void *payload;       /**< Actual data records (casted based on type) */
    
    /**
     * @brief Internal function to free the specialized payload
     */
    void (*free_payload)(void *payload, size_t count);
} DM_Dataset;

/**
 * @brief Core API for dataset management
 */
DM_Dataset* dm_dataset_load(const char *path, DM_DatasetType type);
void dm_dataset_free(DM_Dataset *ds);

#endif // DM_DATASET_H
