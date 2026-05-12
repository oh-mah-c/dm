#ifndef DM_SFU_CE_H
#define DM_SFU_CE_H

#include "core/dm_algorithm.h"

typedef struct {
    int sample_size;
    int max_iterations;
    double quantile;
    double mutation_factor;
} DM_SFU_CE_Params;

#endif
