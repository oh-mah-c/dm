#ifndef DM_FHUQI_MINER_H
#define DM_FHUQI_MINER_H

#include "core/dm_algorithm.h"

typedef enum {
    DM_FHUQI_COMBINE_ALL = 0,
    DM_FHUQI_COMBINE_MIN = 1,
    DM_FHUQI_COMBINE_MAX = 2
} DM_FHUQI_Combine_Method;

typedef struct {
    double min_utility;
    double qrc;
    DM_FHUQI_Combine_Method combine_method;
    const char *profit_path;
} DM_FHUQI_Miner_Params;

#endif
