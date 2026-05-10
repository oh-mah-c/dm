#ifndef DM_ALGORITHM_DEFME_H
#define DM_ALGORITHM_DEFME_H

#include "core/dm_algorithm.h"

/**
 * dEFME Algorithm for depth-first minimal pattern mining (free itemsets).
 * Reference: Arnaud Soulet and François Rioult, "Efficiently Depth-First Minimal Pattern Mining", PAKDD 2014.
 */

typedef struct {
    double min_support;
} DM_DEFME_Params;

#endif // DM_ALGORITHM_DEFME_H
