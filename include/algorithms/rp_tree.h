#ifndef DM_ALGORITHM_RP_TREE_H
#define DM_ALGORITHM_RP_TREE_H

#include "core/dm_algorithm.h"

/**
 * RP-Tree Algorithm for mining rare-item itemsets.
 * Reference: S. Tsang, Y. S. Koh, and G. Dobbie, "RP-Tree: Rare Pattern Tree Mining", DaWaK 2011.
 */

typedef struct {
    double min_freq_support;
    double min_rare_support;
} DM_RP_TREE_Params;

#endif // DM_ALGORITHM_RP_TREE_H
