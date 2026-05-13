#ifndef DM_HUIM_AFSA_H
#define DM_HUIM_AFSA_H

#include "core/dm_algorithm.h"
#include "core/dm_bitset.h"

typedef struct {
    double min_utility;
    int pop_size;
    int max_iter;
    int visual;      // Visual range (Hamming distance)
    int step;        // Movement step size
    int try_number;  // Number of attempts for prey behavior
    double delta;    // Crowd factor
} DM_HUIM_AFSA_Params;

#endif
