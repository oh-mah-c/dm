#ifndef DM_EFIM_H
#define DM_EFIM_H

#include "core/dm_algorithm.h"

/**
 * @file efim.h
 * @brief EFIM: A Highly Efficient Algorithm for High-Utility Itemset Mining
 *
 * Reference:
 *   S. Zida, P. Fournier-Viger, J. C.-W. Lin, C.-W. Wu, and V. S. Tseng,
 *   "EFIM: a highly efficient algorithm for mining high-utility itemsets,"
 *   Knowledge and Information Systems, vol. 51, no. 2, pp. 595-609, 2017.
 *   DOI: 10.1007/s10115-016-0986-0
 *
 * Key features:
 *   - Fast horizontal database projection and transaction merging in linear time.
 *   - Subtree Utility (SU) and Local Utility (LU) upper bounds for aggressive pruning.
 *   - High-performance flat array structures without FP-trees or vertical utility lists.
 */

typedef struct {
    double min_utility; /**< Minimum utility threshold (absolute) */
} DM_EFIM_Params;

#endif // DM_EFIM_H

