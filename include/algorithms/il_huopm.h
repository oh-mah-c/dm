#ifndef DM_IL_HUOPM_H
#define DM_IL_HUOPM_H

#include "core/dm_algorithm.h"

/**
 * @file il_huopm.h
 * @brief IL-HUOPM / HUOMIL: High Utility Occupancy Pattern Mining with Indexed List Structure
 *
 * Reference:
 *   Hyeonmo Kim, Taewoong Ryu, Chanhee Lee, Sinyoung Kim, Bay Vo, Jerry Chun-Wei Lin, and Unil Yun,
 *   "Efficient Method for Mining High Utility Occupancy Patterns based on Indexed List Structure,"
 *   IEEE Access, vol. 11, pp. 45053-45071, May 2023.
 *   DOI: 10.1109/ACCESS.2023.3271864
 *
 * Key Concepts & Data Structures:
 *   - GUO-IL (Global Utility Occupancy Indexed List):
 *       Stores quadruple entries <next_item, uo, ruo, next_idx> for 1-itemsets.
 *   - CUO-IL (Conditional Utility Occupancy Indexed List):
 *       Stores quadruple entries for conditional projected itemsets without two-way list merges.
 *   - Quadruple Entry:
 *       <next_item, uo, ruo, next_idx> maintains direct O(1) index pointers chaining
 *       co-occurring items in each transaction.
 *   - Loose Upper Bound Utility Occupancy (lubuo):
 *       lubuo(P) = (sumUO(P) + sumRUO(P)) / ceil(minSup)
 *       Prunes subtrees in O(1) without requiring top-k sorting.
 *   - Upper Bound Utility Occupancy (ubuo):
 *       ubuo(P) = (1 / k) * sum(top k values of {uo + ruo}), where k = ceil(minSup).
 */

#ifndef DM_PATTERN_CALLBACK_DEFINED
#define DM_PATTERN_CALLBACK_DEFINED
typedef void (*DM_PatternCallback)(const uint32_t *items, size_t length, size_t support, double utility, double occupancy, void *user_data);
#endif

typedef struct {
    double min_occupancy; /**< Minimum utility occupancy threshold minUtilOcc / delta (0.0 to 1.0) */
    double min_support;   /**< Minimum support threshold minSup / gamma (ratio in (0, 1] or absolute count) */
    DM_PatternCallback on_pattern; /**< Optional callback for pluggable post-filters or collectors */
    void *user_data;               /**< User context passed to on_pattern */
} DM_IL_HUOPM_Params;

typedef DM_IL_HUOPM_Params DM_HUOMIL_Params;

#endif // DM_IL_HUOPM_H
