#ifndef DM_ALGORITHM_DIC_H
#define DM_ALGORITHM_DIC_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_support;
    uint32_t block_size; // M transactions per stop
} DM_DIC_Params;

#endif // DM_ALGORITHM_DIC_H
