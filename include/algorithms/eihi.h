#ifndef DM_EIHI_H
#define DM_EIHI_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    int batch_size; // How many transactions to treat as 'N' (increment)
} DM_EIHI_Params;

#endif
