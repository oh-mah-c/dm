#ifndef THUE_H
#define THUE_H

#include "core/dm_algorithm.h"
#include <stddef.h>

typedef struct {
    size_t k;       // Desired number of top-k HUEs
    size_t MTD;     // Maximum Time Duration
} DM_THUE_Params;

typedef struct {
    size_t transactions;
    size_t items_processed;
    size_t candidate_episodes_generated;
    size_t final_hues;
    double final_min_util;
    double initial_min_util;
    double total_utility;
    int timeout_or_memory_limit;
} DM_THUE_Stats;

extern DM_Algorithm thue_algo;

#endif // THUE_H
