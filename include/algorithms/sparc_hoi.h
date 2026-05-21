#ifndef DM_ALGORITHM_SPARC_HOI_H
#define DM_ALGORITHM_SPARC_HOI_H

#include "core/dm_algorithm.h"
#include <stddef.h>

typedef struct {
    double min_occupancy;
    size_t min_support;
    size_t max_patterns;
    double max_seconds;
    int summed_occupancy_mode;
} DM_SPARC_HOI_Params;

extern DM_Algorithm sparc_hoi_algo;

#endif
