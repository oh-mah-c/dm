#ifndef DM_ALGORITHM_LCMVER2_H
#define DM_ALGORITHM_LCMVER2_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_support;
    int mode; // 0: ALL, 1: CLOSED, 2: MAXIMAL
} DM_LCMVER2_Params;

#endif // DM_ALGORITHM_LCMVER2_H
