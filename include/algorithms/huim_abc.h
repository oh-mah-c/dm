#ifndef DM_HUIM_ABC_H
#define DM_HUIM_ABC_H

#include "core/dm_algorithm.h"
#include "core/dm_bitset.h"

typedef struct {
    double min_utility;
    int pop_size;      // Number of food sources (SN)
    int max_iter;
    int limit;         // Trial limit for scout bee
    int step;          // Number of bits to flip in neighborhood search
} DM_HUIM_ABC_Params;

#endif
