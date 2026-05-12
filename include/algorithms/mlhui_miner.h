#ifndef DM_MLHUI_MINER_H
#define DM_MLHUI_MINER_H

#include "core/dm_common.h"
#include "core/dm_dataset.h"
#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    const char *taxonomy_path;
} DM_MLHUI_Miner_Params;

#endif // DM_MLHUI_MINER_H
