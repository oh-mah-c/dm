#ifndef DM_ALGORITHM_CLOSTREAM_H
#define DM_ALGORITHM_CLOSTREAM_H

#include "core/dm_algorithm.h"

/**
 * CloStream Algorithm for maintaining frequent closed itemsets over data streams.
 * Reference: S.-J. Yen et al., "An Efficient Algorithm for Maintaining 
 * Frequent Closed Itemsets over Data Stream", IEA/AIE 2009.
 */

typedef struct {
    double min_support;
} DM_CLOSTREAM_Params;

#endif // DM_ALGORITHM_CLOSTREAM_H
