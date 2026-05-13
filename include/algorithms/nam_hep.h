#ifndef DM_NAM_HEP_H
#define DM_NAM_HEP_H

#include "core/dm_algorithm.h"

typedef struct {
    double support_threshold;   // If -1, use adaptive (median)
    double occupancy_threshold; // If -1, use adaptive (median)
} DM_NAM_HEP_Params;

extern DM_Algorithm nam_hep_algo;

#endif // DM_NAM_HEP_H
