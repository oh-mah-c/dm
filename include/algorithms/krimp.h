#ifndef DM_KRIMP_H
#define DM_KRIMP_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the KRIMP Algorithm
 */
typedef struct {
    double min_support;  // Support threshold for candidate generation
    bool prune;         // Whether to use post-acceptance pruning
} DM_KRIMP_Params;

#endif // DM_KRIMP_H
