#ifndef DM_HAUIM_GMU_H
#define DM_HAUIM_GMU_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility_ratio; // delta
} DM_HAUIM_GMU_Params;

extern DM_Algorithm hauim_gmu_algo;

#endif // DM_HAUIM_GMU_H
