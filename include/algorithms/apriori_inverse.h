#ifndef DM_ALGORITHM_APRIORI_INVERSE_H
#define DM_ALGORITHM_APRIORI_INVERSE_H

#include "core/dm_algorithm.h"

/**
 * Apriori-Inverse Algorithm for finding perfectly sporadic itemsets.
 * Reference: Y. S. Koh and N. Rountree, "Finding Sporadic Rules Using Apriori-Inverse", PAKDD 2005.
 */

typedef struct {
    double max_support;
    uint32_t min_abs_support;
} DM_APRIORI_INVERSE_Params;

#endif // DM_ALGORITHM_APRIORI_INVERSE_H
