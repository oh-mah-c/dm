#ifndef DM_HUIM_SA_H
#define DM_HUIM_SA_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    double temp;
    double min_temp;
    double alpha;
    int pop_size;
} DM_HUIM_SA_Params;

#endif
