#ifndef DM_ALGORITHM_APRIORI_RARE_H
#define DM_ALGORITHM_APRIORI_RARE_H

#include "core/dm_algorithm.h"

/**
 * Apriori-Rare Algorithm for mining minimal rare itemsets (mRIs).
 * Reference: L. Szathmary et al., "Towards Rare Itemset Mining", ICTAI 2007.
 */

typedef struct {
    double min_support;
} DM_APRIORI_RARE_Params;

#endif // DM_ALGORITHM_APRIORI_RARE_H
