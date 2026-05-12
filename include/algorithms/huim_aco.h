#ifndef DM_HUIM_ACO_H
#define DM_HUIM_ACO_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    int pop_size;      // SN
    int max_iter;
    double alpha;      // Local influence
    double beta;       // Global influence
    double gamma;      // Transaction influence
    double lambda;     // Evaporation factor
    double tau;        // Selection threshold
} DM_HUIM_ACO_Params;

#endif
