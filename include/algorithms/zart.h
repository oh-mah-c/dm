#ifndef DM_ALGORITHM_ZART_H
#define DM_ALGORITHM_ZART_H

#include "core/dm_algorithm.h"

/**
 * ZART Algorithm for mining frequent itemsets, closed itemsets and associating generators.
 * Reference: L. Szathmary et al., "ZART: A Multifunctional Itemset Mining Algorithm", CLA 2007.
 */

typedef struct {
    double min_support;
} DM_ZART_Params;

#endif // DM_ALGORITHM_ZART_H
