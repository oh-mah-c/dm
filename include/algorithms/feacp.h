#ifndef DM_FEACP_H
#define DM_FEACP_H

#include "core/dm_dataset_types.h"

typedef struct {
    double min_utility;
    char taxonomy_path[1024];
} DM_FEACP_Params;

#endif
