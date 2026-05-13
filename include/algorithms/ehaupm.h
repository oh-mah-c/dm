#ifndef DM_EHAUPM_H
#define DM_EHAUPM_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility_ratio; // delta (e.g., 0.16 for 16%)
} DM_EHAUPM_Params;

extern DM_Algorithm ehaupm_algo;

#endif // DM_EHAUPM_H
