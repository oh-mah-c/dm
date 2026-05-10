#ifndef DM_ULB_MINER_H
#define DM_ULB_MINER_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    size_t buffer_size; // Optional: specify buffer size
} DM_ULB_Miner_Params;

#endif // DM_ULB_MINER_H
