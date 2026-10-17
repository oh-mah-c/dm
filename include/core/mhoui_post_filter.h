#ifndef DM_MHOUI_POST_FILTER_H
#define DM_MHOUI_POST_FILTER_H

#include <stddef.h>
#include <stdint.h>

/**
 * @file mhoui_post_filter.h
 * @brief Pluggable Post-Filter for Maximal High Utility Occupancy Itemsets (MHOUI)
 *
 * Implements Weak and Strong Pareto Superset Dominance filtering:
 * Given a collection of candidate HOUI / HUOP patterns from any baseline miner
 * (OCEAN, HUOPM, IL-HUOPM, HOUI-Miner, etc.), filters them down to the
 * concise Pareto-maximal frontier without altering the baseline's core mining logic.
 *
 * Dominance Rules:
 *   - Weak Superset Dominance (Definition 5):
 *       Y dominates X if X \subset Y, occ(Y) >= occ(X), util(Y) >= util(X),
 *       and at least one inequality is strict.
 *   - Strong Superset Dominance (Definition 6):
 *       Y dominates X if X \subset Y, occ(Y) >= occ(X), and util(Y) >= util(X).
 */

typedef struct {
    uint32_t *items;
    size_t length;
    size_t support;
    double utility;
    double occupancy;
    uint64_t mask; /**< 64-bit bloom signature for O(1) subset rejection */
} MHOUI_Pattern;

typedef struct {
    MHOUI_Pattern *patterns;
    size_t count;
    size_t capacity;
} MHOUI_PatternSet;

void mhoui_pattern_set_init(MHOUI_PatternSet *set);
void mhoui_pattern_set_free(MHOUI_PatternSet *set);
void mhoui_pattern_set_add(MHOUI_PatternSet *set, const uint32_t *items, size_t length,
                           size_t support, double utility, double occupancy);

/**
 * @brief Filter a pattern set to obtain Pareto-maximal patterns (Weak or Strong)
 *
 * @param input Set of candidate HOUI patterns
 * @param strong 1 for Strong Dominance, 0 for Weak Dominance
 * @param min_utility Minimum utility threshold (if pattern utility < min_utility, prune)
 * @param output Output pattern set populated with maximal patterns
 * @param out_removed Number of dominated / unqualified patterns removed
 * @param time_filter_ms Measured runtime of the post-filtering phase in milliseconds
 * @return 0 on success
 */
int mhoui_post_filter(const MHOUI_PatternSet *input, int strong, double min_utility,
                      MHOUI_PatternSet *output, size_t *out_removed, double *time_filter_ms);

#endif // DM_MHOUI_POST_FILTER_H
