#ifndef DM_TKU_CE_H
#define DM_TKU_CE_H

#include "core/dm_algorithm.h"

typedef struct {
    int k;
    int n;          // Sample number (N)
    double rho;     // Quantile parameter
    int max_iter;
} DM_TKU_CE_Params;

#endif
