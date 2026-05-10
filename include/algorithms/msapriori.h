#ifndef DM_MSAPRIORI_H
#define DM_MSAPRIORI_H

#include "core/dm_common.h"
#include "core/dm_algorithm.h"

/**
 * MSApriori (Multiple Minimum Support Apriori) Parameters.
 * 
 * If mis_values is NULL, the algorithm uses the formula:
 * MIS(i) = max(beta * actual_support(i), LS)
 */
typedef struct {
    double beta;        // Scaling factor (0 <= beta <= 1)
    double LS;          // Minimum support limit (user-specified lowest minsup)
    double *mis_values; // Optional: specific MIS for each item (indexed by item ID)
} DM_MSAPRIORI_Params;

#endif // DM_MSAPRIORI_H
