#ifndef DM_CHUIMINE_H
#define DM_CHUIMINE_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    bool find_maximal; // If true, mine Maximal HUIs. If false, mine Closed+ HUIs.
} DM_CHUIMINE_Params;

#endif
