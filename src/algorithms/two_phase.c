#include "algorithms/two_phase.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t *items;
    size_t count;
    double utility; // TWU in Phase 1, Actual Utility in Phase 2
} TP_Itemset;

typedef struct {
    TP_Itemset **itemsets;
    size_t count;
    size_t capacity;
} TP_Level;

/* --- UTILS --- */

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static TP_Itemset* create_itemset(uint32_t *items, size_t count) {
    TP_Itemset *it = malloc(sizeof(TP_Itemset));
    it->count = count;
    it->items = malloc(sizeof(uint32_t) * count);
    memcpy(it->items, items, sizeof(uint32_t) * count);
    qsort(it->items, count, sizeof(uint32_t), cmp_uint32);
    it->utility = 0;
    return it;
}

static void free_itemset(TP_Itemset *it) {
    free(it->items);
    free(it);
}

static bool is_subset(const uint32_t *sub, size_t sub_len, const DM_Trans_Utility *tr) {
    size_t i = 0, j = 0;
    while (i < sub_len && j < tr->count) {
        if (sub[i] == tr->items[j].id) { i++; j++; }
        else if (sub[i] > tr->items[j].id) j++;
        else return false;
    }
    return i == sub_len;
}

static double calculate_actual_utility(const uint32_t *items, size_t count, const DM_Trans_Utility *tr) {
    double util = 0;
    size_t i = 0, j = 0;
    while (i < count && j < tr->count) {
        if (items[i] == tr->items[j].id) {
            util += tr->items[j].utility;
            i++; j++;
        }
        else if (items[i] > tr->items[j].id) j++;
        else break;
    }
    return util;
}

/* --- CANDIDATE GENERATION (APRIORI STYLE) --- */

static TP_Level* generate_candidates(TP_Level *prev_level) {
    TP_Level *candidates = malloc(sizeof(TP_Level));
    candidates->count = 0;
    candidates->capacity = 100;
    candidates->itemsets = malloc(sizeof(TP_Itemset*) * candidates->capacity);

    size_t k = prev_level->itemsets[0]->count;

    for (size_t i = 0; i < prev_level->count; i++) {
        for (size_t j = i + 1; j < prev_level->count; j++) {
            TP_Itemset *it1 = prev_level->itemsets[i];
            TP_Itemset *it2 = prev_level->itemsets[j];

            // Join step
            bool can_join = true;
            for (size_t l = 0; l < k - 1; l++) {
                if (it1->items[l] != it2->items[l]) { can_join = false; break; }
            }
            if (can_join) {
                uint32_t *new_items = malloc(sizeof(uint32_t) * (k + 1));
                memcpy(new_items, it1->items, sizeof(uint32_t) * k);
                new_items[k] = it2->items[k - 1];
                qsort(new_items, k + 1, sizeof(uint32_t), cmp_uint32);

                // Pruning step (all subsets must be in prev_level)
                bool all_subsets_present = true;
                for (size_t l = 0; l < k + 1; l++) {
                    uint32_t *subset = malloc(sizeof(uint32_t) * k);
                    size_t si = 0;
                    for (size_t m = 0; m < k + 1; m++) {
                        if (m == l) continue;
                        subset[si++] = new_items[m];
                    }
                    
                    bool subset_found = false;
                    for (size_t m = 0; m < prev_level->count; m++) {
                        if (memcmp(subset, prev_level->itemsets[m]->items, sizeof(uint32_t) * k) == 0) {
                            subset_found = true; break;
                        }
                    }
                    free(subset);
                    if (!subset_found) { all_subsets_present = false; break; }
                }

                if (all_subsets_present) {
                    if (candidates->count >= candidates->capacity) {
                        candidates->capacity *= 2;
                        candidates->itemsets = realloc(candidates->itemsets, sizeof(TP_Itemset*) * candidates->capacity);
                    }
                    candidates->itemsets[candidates->count++] = create_itemset(new_items, k + 1);
                }
                free(new_items);
            }
        }
    }
    return candidates;
}

/* --- MAIN RUN --- */

static int cmp_dm_items(const void *a, const void *b) {
    uint32_t id1 = ((DM_Item*)a)->id;
    uint32_t id2 = ((DM_Item*)b)->id;
    return (id1 < id2) ? -1 : ((id1 > id2) ? 1 : 0);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) {
        printf("[Two-Phase] Error: This algorithm requires a utility-based dataset.\n");
        return DM_ERROR_INCOMPATIBLE;
    }

    DM_Two_Phase_Params *p = (DM_Two_Phase_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        qsort(data[i].items, data[i].count, sizeof(DM_Item), cmp_dm_items);
    }

    printf("[Two-Phase] Starting Phase I (TWU Mining)... Min Utility: %.2f\n", min_util);

    TP_Level **all_levels = malloc(sizeof(TP_Level*) * 100);
    size_t level_count = 0;

    // --- L1 ---
    TP_Level *l1 = malloc(sizeof(TP_Level));
    l1->capacity = ds->max_id + 1;
    l1->count = 0;
    l1->itemsets = malloc(sizeof(TP_Itemset*) * l1->capacity);

    double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            twu_counts[data[i].items[j].id] += data[i].total_utility;
        }
    }

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu_counts[i] >= min_util) {
            uint32_t item = i;
            TP_Itemset *it = create_itemset(&item, 1);
            it->utility = twu_counts[i];
            l1->itemsets[l1->count++] = it;
        }
    }
    free(twu_counts);
    all_levels[level_count++] = l1;
    printf("[Two-Phase] L1 found %zu itemsets.\n", l1->count);

    // --- Lk ---
    while (all_levels[level_count - 1]->count > 0) {
        TP_Level *candidates = generate_candidates(all_levels[level_count - 1]);
        if (candidates->count == 0) {
            free(candidates->itemsets); free(candidates);
            break;
        }

        // Count TWU
        for (size_t i = 0; i < ds->count; i++) {
            for (size_t j = 0; j < candidates->count; j++) {
                if (is_subset(candidates->itemsets[j]->items, candidates->itemsets[j]->count, &data[i])) {
                    candidates->itemsets[j]->utility += data[i].total_utility;
                }
            }
        }

        // Filter
        TP_Level *lk = malloc(sizeof(TP_Level));
        lk->capacity = candidates->count;
        lk->count = 0;
        lk->itemsets = malloc(sizeof(TP_Itemset*) * lk->capacity);

        for (size_t j = 0; j < candidates->count; j++) {
            if (candidates->itemsets[j]->utility >= min_util) {
                lk->itemsets[lk->count++] = candidates->itemsets[j];
            } else {
                free_itemset(candidates->itemsets[j]);
            }
        }
        free(candidates->itemsets); free(candidates);

        all_levels[level_count++] = lk;
        printf("[Two-Phase] L%zu found %zu itemsets.\n", level_count, lk->count);
        if (lk->count == 0) break;
    }

    // --- Phase II ---
    printf("[Two-Phase] Starting Phase II (Actual Utility Check)...\n");
    size_t total_candidates = 0;
    for (size_t i = 0; i < level_count; i++) total_candidates += all_levels[i]->count;

    TP_Itemset **candidates = malloc(sizeof(TP_Itemset*) * total_candidates);
    size_t ci = 0;
    for (size_t i = 0; i < level_count; i++) {
        for (size_t j = 0; j < all_levels[i]->count; j++) {
            candidates[ci] = all_levels[i]->itemsets[j];
            candidates[ci]->utility = 0; // Reset for actual utility
            ci++;
        }
    }

    // Scan once for all candidates
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < total_candidates; j++) {
            if (is_subset(candidates[j]->items, candidates[j]->count, &data[i])) {
                candidates[j]->utility += calculate_actual_utility(candidates[j]->items, candidates[j]->count, &data[i]);
            }
        }
    }

    size_t high_utility_count = 0;
    size_t total_items = 0;
    for (size_t j = 0; j < total_candidates; j++) {
        if (candidates[j]->utility >= min_util) {
            high_utility_count++;
            total_items += candidates[j]->count;
            // Optionally print or store
        }
    }

    printf("[Two-Phase] Found %zu High Utility Itemsets.\n", high_utility_count);

    // Cleanup
    for (size_t i = 0; i < total_candidates; i++) free_itemset(candidates[i]);
    free(candidates);
    for (size_t i = 0; i < level_count; i++) {
        free(all_levels[i]->itemsets);
        free(all_levels[i]);
    }
    free(all_levels);

    dm_bench_record_results(high_utility_count, total_items);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "twophase",
    .name = "Two-Phase High Utility Itemset Mining",
    .description = "A Two-Phase algorithm for fast discovery of high utility itemsets using transaction-weighted utilization.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
