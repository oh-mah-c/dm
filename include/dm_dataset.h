#ifndef DM_DATASET_H
#define DM_DATASET_H

#include "dm_common.h"

/**
 * @brief Types of datasets supported by the framework
 */
typedef enum {
    DM_TYPE_TRANSACTIONAL, /**< Simple sets of items (e.g., 1 2 5) */
    DM_TYPE_UTILITY,       /**< Items with utility/quantity (e.g., 1:10 2:5) */
    DM_TYPE_MATRIX         /**< Numeric matrix for clustering/classification */
} DM_DatasetType;

/**
 * @brief Represents an item in a transaction, optionally with utility
 */
typedef struct {
    uint32_t id;
    double utility; 
} DM_Item;

/**
 * @brief Represents a single transaction or data point
 */
typedef struct {
    DM_Item *items;
    size_t count;
    double total_utility; /**< Used for utility mining (e.g., TU in HUI Mining) */
} DM_Transaction;

/**
 * @brief Generic dataset structure
 */
typedef struct {
    DM_DatasetType type;
    DM_Transaction *transactions;
    size_t transaction_count;
    uint32_t max_item_id;
} DM_Dataset;

/**
 * @brief Loads a dataset from a file with a specific type hint
 * @param path Path to the dataset file
 * @param type The expected type of the dataset
 * @return Pointer to the loaded dataset, or NULL on failure
 */
DM_Dataset* dm_dataset_load(const char *path, DM_DatasetType type);

/**
 * @brief Frees the memory associated with a dataset
 * @param ds Pointer to the dataset to free
 */
void dm_dataset_free(DM_Dataset *ds);

#endif // DM_DATASET_H
