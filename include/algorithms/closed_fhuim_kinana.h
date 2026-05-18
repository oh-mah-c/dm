#ifndef DM_CLOSED_FHUIM_KINANA_H
#define DM_CLOSED_FHUIM_KINANA_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    int min_support;
    double min_owl;
} DM_ClosedFHUIMKinana_Params;

extern DM_Algorithm closed_fhuim_kinana_algo;

#endif
