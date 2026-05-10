#ifndef DM_APRIORI_HYBRID_H
#define DM_APRIORI_HYBRID_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the AprioriHybrid Algorithm
 */
typedef struct {
    double min_support;
    double min_confidence;
} DM_APRIORI_HYBRID_Params;

#endif // DM_APRIORI_HYBRID_H
