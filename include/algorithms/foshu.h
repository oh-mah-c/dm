#ifndef DM_FOSHU_H
#define DM_FOSHU_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility; // Threshold between 0 and 1
    int num_periods;    // Number of periods to split the dataset into
} DM_FOSHU_Params;

#endif
