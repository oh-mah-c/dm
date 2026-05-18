#ifndef DM_DPHIM_H
#define DM_DPHIM_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    int threads;
} DM_DPHIM_Params;

extern DM_Algorithm dphim_algo;

#endif
