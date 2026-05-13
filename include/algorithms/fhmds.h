#ifndef DM_FHMDS_H
#define DM_FHMDS_H

#include "core/dm_dataset.h"
#include "core/dm_algorithm.h"

typedef struct {
    int k;
    int batch_size;
    int window_size;
} DM_FHMDS_Params;

extern DM_Algorithm fhmds_algo;

#endif
