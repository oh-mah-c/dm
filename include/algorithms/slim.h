#ifndef DM_SLIM_H
#define DM_SLIM_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the SLIM Algorithm
 */
typedef struct {
    bool prune;         // Whether to use post-acceptance pruning
} DM_SLIM_Params;

#endif // DM_SLIM_H
