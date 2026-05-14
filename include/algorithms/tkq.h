#ifndef DM_TKQ_H
#define DM_TKQ_H

#include "core/dm_algorithm.h"
#include "algorithms/fhuqi_miner.h"

typedef struct {
    size_t k;
    double qrc;
    DM_FHUQI_Combine_Method combine_method;
    const char *profit_path;
} DM_TKQ_Params;

#endif
