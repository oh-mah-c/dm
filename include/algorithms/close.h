#ifndef DM_CLOSE_H
#define DM_CLOSE_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the Close Algorithm
 */
typedef struct {
    double min_support;
    double min_confidence;
} DM_CLOSE_Params;

#endif // DM_CLOSE_H
