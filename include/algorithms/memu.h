#ifndef DM_MEMU_H
#define DM_MEMU_H

#include "core/dm_algorithm.h"

typedef struct {
    double *mau_table; // mau(i) for each item i
    size_t item_count;
} DM_MEMU_Params;

extern DM_Algorithm memu_algo;

#endif // DM_MEMU_H
