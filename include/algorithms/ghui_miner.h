#ifndef DM_GHUI_MINER_H
#define DM_GHUI_MINER_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    bool mine_ghui; // true for GHUI-Miner, false for HUG-Miner
} DM_GHUI_Miner_Params;

#endif
