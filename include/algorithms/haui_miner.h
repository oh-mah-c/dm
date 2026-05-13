#ifndef DM_HAUI_MINER_H
#define DM_HAUI_MINER_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility_ratio; // delta (e.g., 0.16 for 16%)
    double min_utility_threshold; // Calculated as delta * TU
} DM_HAUI_Miner_Params;

extern DM_Algorithm haui_miner_algo;

#endif // DM_HAUI_MINER_H
