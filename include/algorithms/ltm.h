#ifndef DM_ALGORITHM_LTM_H
#define DM_ALGORITHM_LTM_H

#include "core/dm_algorithm.h"

/**
 * Frequent Itemset Mining Algorithm Based on Linear Table (LTM).
 * Reference: Jun Lu, Wenhe Xu, Kailong Zhou, Zhicong Guo,
 * "Frequent Itemset Mining Algorithm Based on Linear Table", 
 * Journal of Database Management, Volume 34, Issue 1, 2023.
 */

typedef struct {
    double min_support;
} DM_LTM_Params;

#endif // DM_ALGORITHM_LTM_H
