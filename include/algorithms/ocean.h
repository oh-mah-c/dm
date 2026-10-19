#ifndef DM_OCEAN_H
#define DM_OCEAN_H

#include "core/dm_algorithm.h"

/**
 * @file ocean.h
 * @brief OCEAN: Fast Discovery of High Utility Occupancy Itemsets
 *
 * Reference:
 *   Bilong Shen, Zhaoduo Wen, Ying Zhao, Dongliang Zhou, and Weimin Zheng,
 *   "OCEAN: Fast Discovery of High Utility Occupancy Itemsets,"
 *   in Proc. 21st International Conference on Database Systems for Advanced Applications (DASFAA 2016),
 *   Springer LNCS 9642, 2016, pp. 355-369.
 *   DOI: 10.1007/978-3-319-32025-0_22
 *
 * Problem Formulation:
 *   - Utility occupancy of itemset W in transaction t:
 *       o_t(W) = u_t(W) / U(t)
 *   - Utility occupancy of W in database D:
 *       O(W) = (1 / |D_W|) * sum_{t in D_W} (u_t(W) / U(t))
 *   - Qualified HUOI condition:
 *       O(W) >= minOccu  AND  |D_W| >= ceil(minSupp * |D|)
 *
 * Pruning:
 *   - UpperBound(PX):
 *       Let V = { (u_t(PX) + u_t(E)) / U(t) | t in D_PX }
 *       maxOccu = (1 / k) * sum(top k values of V), where k = ceil(minSupp * |D|)
 *       If maxOccu < minOccu, prune the entire subtree!
 */

typedef void (*DM_PatternCallback)(const uint32_t *items, size_t length, size_t support, double utility, double occupancy, void *user_data);

typedef struct {
    double min_occupancy; /**< Minimum utility occupancy threshold minOccu (0.0 to 1.0) */
    double min_support;   /**< Minimum support ratio minSupp (e.g. 0.01) or absolute count */
    DM_PatternCallback on_pattern; /**< Optional callback for pluggable post-filters or collectors */
    void *user_data;               /**< User context passed to on_pattern */
} DM_OCEAN_Params;

#endif // DM_OCEAN_H
