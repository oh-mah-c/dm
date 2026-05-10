#ifndef DM_ALGORITHM_CARPENTER_H
#define DM_ALGORITHM_CARPENTER_H

#include "core/dm_algorithm.h"

/**
 * CARPENTER Algorithm for finding closed patterns in long datasets.
 * Reference: Feng Pan et al., "CARPENTER: Finding Closed Patterns in Long Biological Datasets", 2003.
 */

typedef struct {
    double min_support;
} DM_CARPENTER_Params;

#endif // DM_ALGORITHM_CARPENTER_H
