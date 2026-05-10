#ifndef DM_APRIORI_H
#define DM_APRIORI_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the Apriori Algorithm
 */
typedef struct {
    double min_support;
    double min_confidence;
} DM_APRIORI_Params;

#endif // DM_APRIORI_H
