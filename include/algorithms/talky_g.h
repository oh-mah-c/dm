#ifndef DM_ALGORITHM_TALKY_G_H
#define DM_ALGORITHM_TALKY_G_H

#include "core/dm_algorithm.h"

/**
 * Talky-G Algorithm for vertical mining of frequent generators.
 * Reference: Laszlo Szathmary et al., "Efficient Vertical Mining of Frequent Closures and Generators", IDA 2009.
 */

typedef struct {
    double min_support;
} DM_TALKY_G_Params;

#endif // DM_ALGORITHM_TALKY_G_H
