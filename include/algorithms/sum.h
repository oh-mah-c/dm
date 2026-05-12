#ifndef DM_SUM_H
#define DM_SUM_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;    /* MU_0 */
    int window_size;       /* W_n */
    int increment_size;    /* Size of each batch for incremental simulation */
    bool dynamic_threshold; /* Whether to raise threshold dynamically */
} DM_SUM_Params;

#endif // DM_SUM_H
