#ifndef DM_SKYLINE_MINER_H
#define DM_SKYLINE_MINER_H

#include "core/dm_algorithm.h"

typedef struct {
    // Skyline Miner typically doesn't require min_util or min_supp,
    // but we can provide optional ones to limit the search space.
    double min_utility;
    int min_support;
} DM_Skyline_Miner_Params;

#endif
