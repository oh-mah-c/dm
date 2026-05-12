#ifndef DM_HUIM_BPSO_TREE_H
#define DM_HUIM_BPSO_TREE_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    int pop_size;
    int max_iter;
    double w;
    double c1;
    double c2;
} DM_HUIM_BPSO_Tree_Params;

#endif
