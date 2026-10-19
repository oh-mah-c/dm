/**
 * @file mhoui_post_filter.c
 * @brief Modular Pluggable Post-Filter for Maximal High Utility Occupancy Itemsets
 *
 * Implements Weak and Strong Pareto Superset Dominance checking over
 * candidate patterns discovered by two-stage baselines (OCEAN, HUOPM, IL-HUOPM, HOUI-Miner).
 *
 * Fully decoupled from baseline mining logic:
 *   - 64-bit Bloom signatures for O(1) non-subset rejection.
 *   - Descending-length sort so candidates are only tested against longer maximal patterns.
 *   - Independent high-resolution runtime and memory profiling.
 */

#include "core/mhoui_post_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MHOUI_EPS 1e-12

static inline uint64_t compute_mask(const uint32_t *items, size_t length) {
    uint64_t mask = 0;
    for (size_t i = 0; i < length; i++) {
        mask |= UINT64_C(1) << (items[i] & 63);
    }
    return mask;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    return (ua < ub) ? -1 : (ua > ub) ? 1 : 0;
}

void mhoui_pattern_set_init(MHOUI_PatternSet *set) {
    if (!set) return;
    set->patterns = NULL;
    set->count = 0;
    set->capacity = 0;
}

void mhoui_pattern_set_free(MHOUI_PatternSet *set) {
    if (!set) return;
    if (set->patterns) {
        for (size_t i = 0; i < set->count; i++) {
            free(set->patterns[i].items);
        }
        free(set->patterns);
        set->patterns = NULL;
    }
    set->count = 0;
    set->capacity = 0;
}

void mhoui_pattern_set_add(MHOUI_PatternSet *set, const uint32_t *items, size_t length,
                           size_t support, double utility, double occupancy) {
    if (!set || !items || length == 0) return;

    if (set->count >= set->capacity) {
        size_t new_cap = set->capacity == 0 ? 128 : set->capacity * 2;
        MHOUI_Pattern *new_pats = (MHOUI_Pattern *)realloc(set->patterns, new_cap * sizeof(MHOUI_Pattern));
        if (!new_pats) return;
        set->patterns = new_pats;
        set->capacity = new_cap;
    }

    MHOUI_Pattern *p = &set->patterns[set->count];
    p->length = length;
    p->support = support;
    p->utility = utility;
    p->occupancy = occupancy;
    p->items = (uint32_t *)malloc(length * sizeof(uint32_t));
    if (!p->items) return;
    memcpy(p->items, items, length * sizeof(uint32_t));
    qsort(p->items, length, sizeof(uint32_t), cmp_uint32);
    p->mask = compute_mask(p->items, length);

    set->count++;
}

/* Sort descending by length; if tied, descending by support, then utility */
static int cmp_pattern_desc(const void *a, const void *b) {
    const MHOUI_Pattern *pa = (const MHOUI_Pattern *)a;
    const MHOUI_Pattern *pb = (const MHOUI_Pattern *)b;
    if (pa->length != pb->length) {
        return (pa->length > pb->length) ? -1 : 1;
    }
    if (pa->support != pb->support) {
        return (pa->support > pb->support) ? -1 : 1;
    }
    if (pa->utility > pb->utility + MHOUI_EPS) return -1;
    if (pa->utility + MHOUI_EPS < pb->utility) return 1;
    return 0;
}

/**
 * @brief Checks if x is a proper subset of y: x \subset y.
 * Assumes both x->items and y->items are sorted in ascending order.
 */
static inline int is_proper_subset(const MHOUI_Pattern *x, const MHOUI_Pattern *y) {
    if (x->length >= y->length) return 0;
    /* Bloom signature quick check: all 1-bits in x must be in y */
    if ((x->mask & ~y->mask) != 0) return 0;

    size_t i = 0, j = 0;
    while (i < x->length && j < y->length) {
        if (x->items[i] == y->items[j]) {
            i++;
            j++;
        } else if (x->items[i] > y->items[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return i == x->length;
}

/**
 * @brief Checks if x and y represent the identical itemset.
 */
static inline int is_identical(const MHOUI_Pattern *x, const MHOUI_Pattern *y) {
    if (x->length != y->length) return 0;
    if (x->mask != y->mask) return 0;
    return memcmp(x->items, y->items, x->length * sizeof(uint32_t)) == 0;
}

int mhoui_post_filter(const MHOUI_PatternSet *input, int strong, double min_utility,
                      MHOUI_PatternSet *output, size_t *out_removed, double *time_filter_ms) {
    if (!input || !output) return -1;

    struct timespec start_ts, end_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    mhoui_pattern_set_init(output);
    size_t removed = 0;

    if (input->count == 0) {
        if (out_removed) *out_removed = 0;
        if (time_filter_ms) *time_filter_ms = 0.0;
        return 0;
    }

    /* 1. Filter out patterns that do not satisfy min_utility */
    MHOUI_Pattern *candidates = (MHOUI_Pattern *)malloc(input->count * sizeof(MHOUI_Pattern));
    if (!candidates) return -1;

    size_t cand_count = 0;
    for (size_t i = 0; i < input->count; i++) {
        if (input->patterns[i].utility + MHOUI_EPS >= min_utility) {
            candidates[cand_count++] = input->patterns[i];
        } else {
            removed++;
        }
    }

    /* 2. Sort candidate patterns in descending order of length */
    qsort(candidates, cand_count, sizeof(MHOUI_Pattern), cmp_pattern_desc);

    /* 3. Progressive Pareto dominance check:
     *    Because candidates are sorted in descending length order,
     *    longer patterns are evaluated and accepted first.
     *    A candidate x can only be dominated by an already-accepted maximal pattern y (length(y) >= length(x)).
     */
    for (size_t i = 0; i < cand_count; i++) {
        const MHOUI_Pattern *x = &candidates[i];
        int dominated = 0;

        for (size_t j = 0; j < output->count; j++) {
            const MHOUI_Pattern *y = &output->patterns[j];

            /* Deduplicate identical itemsets */
            if (is_identical(x, y)) {
                dominated = 1;
                break;
            }

            if (!is_proper_subset(x, y)) {
                continue;
            }

            /* Superset Pareto dominance conditions */
            int ge_occ = (y->occupancy + MHOUI_EPS >= x->occupancy);
            int ge_util = (y->utility + MHOUI_EPS >= x->utility);

            if (strong) {
                /* Strong Dominance (Def 6): y dominates x if occ(y) >= occ(x) and util(y) >= util(x) */
                if (ge_occ && ge_util) {
                    dominated = 1;
                    break;
                }
            } else {
                /* Weak Dominance (Def 5): y dominates x if occ(y) >= occ(x), util(y) >= util(x),
                 * and at least one inequality is strict. */
                int strict = (y->occupancy > x->occupancy + MHOUI_EPS) ||
                             (y->utility > x->utility + MHOUI_EPS);
                if (ge_occ && ge_util && strict) {
                    dominated = 1;
                    break;
                }
            }
        }

        if (dominated) {
            removed++;
        } else {
            mhoui_pattern_set_add(output, x->items, x->length, x->support, x->utility, x->occupancy);
        }
    }

    free(candidates);

    clock_gettime(CLOCK_MONOTONIC, &end_ts);
    double elapsed_ms = (double)(end_ts.tv_sec - start_ts.tv_sec) * 1000.0 +
                        (double)(end_ts.tv_nsec - start_ts.tv_nsec) / 1000000.0;

    if (out_removed) *out_removed = removed;
    if (time_filter_ms) *time_filter_ms = elapsed_ms;

    return 0;
}
