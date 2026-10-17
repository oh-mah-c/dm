#ifndef DM_HUOPM_H
#define DM_HUOPM_H

#include "core/dm_algorithm.h"

/**
 * @file huopm.h
 * @brief HUOPM: High-Utility Occupancy Pattern Mining
 *
 * Reference:
 *   Wensheng Gan, Jerry Chun-Wei Lin, Philippe Fournier-Viger, Han-Chieh Chao, and Philip S. Yu,
 *   "HUOPM: High-Utility Occupancy Pattern Mining,"
 *   IEEE Transactions on Cybernetics, vol. 50, no. 3, pp. 1195-1208, March 2020.
 *   DOI: 10.1109/TCYB.2019.2894367
 *
 * Key Concepts & Properties:
 *   - Relative Utility Occupancy of pattern X in transaction t:
 *       uo(X, t) = u(X, t) / TU(t)
 *   - Remaining Utility Occupancy of pattern X in transaction t:
 *       ruo(X, t) = ru(X, t) / TU(t) = (sum_{y > X, y in t} u(y, t)) / TU(t)
 *   - Average Utility Occupancy:
 *       uo(X) = (1 / |X.UOL|) * sum_{t in X.UOL} uo(X, t)
 *   - Qualified HUOP:
 *       uo(X) >= beta (minOccu)  AND  sup(X) >= ceil(alpha * |D|) (minSupp)
 *
 * Pruning Strategies (Gan et al., IEEE TCYB 2020):
 *   - Strategy 1 (Support Pruning):
 *       If sup(X) < ceil(alpha * |D|), prune X and all supersets (Global Downward Closure).
 *   - Strategy 2 (Utility Occupancy Upper Bound):
 *       UpperBound(X, alpha) = (1 / k) * sum(top k values of {uo(X, t) + ruo(X, t)}),
 *       where k = ceil(alpha * |D|).
 *       If UpperBound(X, alpha) < beta, prune the entire subtree rooted at X (Partial Downward Closure).
 *   - Strategy 3 (Early Construction Abandonment):
 *       In Construct(X, Xa, Xb), if matched_count + remaining_elements(Xa) < k,
 *       abandon construction immediately and return NULL.
 *   - Strategy 4 (Infrequent Child Pruning):
 *       If sup(Xab) < k or Xab.UOL is empty, discard Xab from extenOfXa.
 */

#ifndef DM_PATTERN_CALLBACK_DEFINED
#define DM_PATTERN_CALLBACK_DEFINED
typedef void (*DM_PatternCallback)(const uint32_t *items, size_t length, size_t support, double utility, double occupancy, void *user_data);
#endif

typedef struct {
    double min_occupancy; /**< Minimum utility occupancy threshold beta (0.0 to 1.0) */
    double min_support;   /**< Minimum support threshold alpha (ratio in (0, 1] or absolute count) */
    DM_PatternCallback on_pattern; /**< Optional callback for pluggable post-filters or collectors */
    void *user_data;               /**< User context passed to on_pattern */
} DM_HUOPM_Params;

#endif // DM_HUOPM_H
