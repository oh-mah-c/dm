#ifndef DM_HUP_MINER_H
#define DM_HUP_MINER_H

#include "core/dm_algorithm.h"
#include <stddef.h>

typedef struct {
    double min_average_utility;
    size_t max_pattern_length;   /* 0 means continue until no HUBPs remain. */
    size_t max_candidates;       /* 0 means no explicit candidate cap. */
} DM_HUP_Miner_Params;

extern DM_Algorithm hup_miner_algo;

#endif
