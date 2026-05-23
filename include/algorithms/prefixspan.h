#ifndef DM_PREFIXSPAN_H
#define DM_PREFIXSPAN_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for PrefixSpan
 */
typedef struct {
    double min_support;       // Minimum support (if < 1.0, it's a ratio, else count)
} DM_PrefixSpan_Params;

#endif // DM_PREFIXSPAN_H
