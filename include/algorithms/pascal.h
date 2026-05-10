#ifndef DM_ALGORITHM_PASCAL_H
#define DM_ALGORITHM_PASCAL_H

#include "core/dm_algorithm.h"

/**
 * PASCAL Algorithm for mining frequent patterns using counting inference.
 * Reference: N. Pasquier et al., "Mining Frequent Patterns with Counting Inference", SIGKDD Explorations, 2000.
 */

typedef struct {
    double min_support;
} DM_PASCAL_Params;

#endif // DM_ALGORITHM_PASCAL_H
