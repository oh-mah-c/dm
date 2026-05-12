#ifndef DM_HUPE_GARM_H
#define DM_HUPE_GARM_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    int pop_size;
    int max_iter;
    double p_max;
    double p_min;
    int k;
} DM_HUPE_GARM_Params;

#endif
