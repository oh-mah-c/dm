#ifndef DM_ALGORITHM_CORI_H
#define DM_ALGORITHM_CORI_H

#include "core/dm_algorithm.h"

/**
 * CORI Algorithm for mining rare correlated patterns using the bond measure.
 * Reference: S. Bouasker and S. Ben Yahia, "Key correlation mining by simultaneous 
 * monotone and anti-monotone constraints checking", SAC 2015.
 */

typedef struct {
    double min_support;
    double min_bond;
} DM_CORI_Params;

#endif // DM_ALGORITHM_CORI_H
