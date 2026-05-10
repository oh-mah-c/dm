#ifndef DM_ALGORITHM_DBV_MINER_H
#define DM_ALGORITHM_DBV_MINER_H

#include "core/dm_algorithm.h"

/**
 * DBV-Miner Algorithm for fast mining frequent closed itemsets.
 * Reference: Bay Vo, Tzung-Pei Hong, Bac Le, "DBV-Miner: A Dynamic Bit-Vector approach for fast mining frequent closed itemsets", 2012.
 */

typedef struct {
    double min_support;
} DM_DBV_MINER_Params;

#endif // DM_ALGORITHM_DBV_MINER_H
