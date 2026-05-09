#ifndef DM_AIS_H
#define DM_AIS_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the AIS Algorithm
 */
typedef struct {
    double min_support; /**< Minimum support threshold (e.g., 0.01 for 1% or 10.0 for 10 absolute) */
} DM_AIS_Params;

#endif // DM_AIS_H
