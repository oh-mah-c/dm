#ifndef DM_APRIORI_TID_H
#define DM_APRIORI_TID_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the AprioriTid Algorithm
 */
typedef struct {
    double min_support;
    double min_confidence;
} DM_APRIORI_TID_Params;

#endif // DM_APRIORI_TID_H
