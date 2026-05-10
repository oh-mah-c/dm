#ifndef DM_ALGORITHM_SAM_H
#define DM_ALGORITHM_SAM_H

#include "core/dm_algorithm.h"

/**
 * SaM (Split and Merge) Algorithm.
 * Reference: Christian Borgelt and Xiaomeng Wang, 
 * "SaM: A Split and Merge Algorithm for Fuzzy Frequent Item Set Mining", 2009.
 */

typedef struct {
    double min_support;
} DM_SAM_Params;

#endif // DM_ALGORITHM_SAM_H
