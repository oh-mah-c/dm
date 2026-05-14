#ifndef DM_DATASET_TYPES_H
#define DM_DATASET_TYPES_H

#include "dm_common.h"

/**
 * @brief Item structure for utility-based mining
 */
typedef struct {
    uint32_t id;
    double utility;
} DM_Item;

/**
 * @brief Simple transaction (List of IDs)
 * Used by: Apriori, Eclat, etc.
 */
typedef struct {
    uint32_t *items;
    size_t count;
} DM_Trans_Simple;

/**
 * @brief Utility transaction (Items with individual utilities + Total Utility)
 * Used by: HUI-Mining, FHN, etc.
 */
typedef struct {
    DM_Item *items;
    size_t count;
    double total_utility;
} DM_Trans_Utility;

/**
 * @brief Quantity transaction (Items with purchased quantities)
 * Used by: HUQIM algorithms requiring separate quantities and external utilities.
 */
typedef struct {
    uint32_t id;
    double quantity;
} DM_Quantity_Item;

typedef struct {
    DM_Quantity_Item *items;
    size_t count;
} DM_Trans_Quantity;

/**
 * @brief Matrix row (Array of doubles)
 * Used by: Clustering, Classification, PCA, etc.
 */
typedef struct {
    double *values;
    size_t count;
} DM_Matrix_Row;

/**
 * @brief Utility itemset in a sequence
 */
typedef struct {
    DM_Item *items;
    size_t count;
} DM_Trans_Sequence_Utility;

/**
 * @brief High utility sequence (Sequence of itemsets)
 */
typedef struct {
    DM_Trans_Sequence_Utility *itemsets;
    size_t count;
    double total_utility;
    double probability; /**< Probability of this sequence (for uncertain data) */
} DM_Sequence_Utility;

#endif // DM_DATASET_TYPES_H
