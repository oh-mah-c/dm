#include "algorithms/apriori_inverse.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Apriori-Inverse Algorithm for finding perfectly sporadic itemsets.
 * Reference: Y. S. Koh and N. Rountree, "Finding Sporadic Rules Using Apriori-Inverse", PAKDD 2005.
 */

typedef struct {
    uint32_t *items;
    size_t len;
    uint32_t *tids;
    size_t tid_count;
} IVNode;

typedef struct {
    IVNode *nodes;
    size_t count;
    size_t capacity;
} IVLevel;

static void add_ivnode(IVLevel *lvl, uint32_t *items, size_t len, uint32_t *tids, size_t tid_count) {
    if (lvl->count == lvl->capacity) {
        lvl->capacity = lvl->capacity == 0 ? 16 : lvl->capacity * 2;
        lvl->nodes = realloc(lvl->nodes, lvl->capacity * sizeof(IVNode));
    }
    lvl->nodes[lvl->count].items = malloc(len * sizeof(uint32_t));
    memcpy(lvl->nodes[lvl->count].items, items, len * sizeof(uint32_t));
    lvl->nodes[lvl->count].len = len;
    lvl->nodes[lvl->count].tids = malloc(tid_count * sizeof(uint32_t));
    memcpy(lvl->nodes[lvl->count].tids, tids, tid_count * sizeof(uint32_t));
    lvl->nodes[lvl->count].tid_count = tid_count;
    lvl->count++;
}

static void free_level(IVLevel *lvl) {
    for (size_t i = 0; i < lvl->count; i++) {
        free(lvl->nodes[i].items);
        free(lvl->nodes[i].tids);
    }
    free(lvl->nodes);
    lvl->nodes = NULL;
    lvl->count = 0;
    lvl->capacity = 0;
}

static size_t intersect(const uint32_t *t1, size_t n1, const uint32_t *t2, size_t n2, uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (t1[i] < t2[j]) i++;
        else if (t1[i] > t2[j]) j++;
        else {
            if (out) out[k] = t1[i];
            i++; j++; k++;
        }
    }
    return k;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_APRIORI_INVERSE_Params *p = (DM_APRIORI_INVERSE_Params *)params;
    double max_sup_param = p ? p->max_support : 0.25;
    uint32_t max_sup = (uint32_t)ceil(max_sup_param * ds->count);
    uint32_t min_abs_sup = p ? p->min_abs_support : 5;

    printf("[Apriori-Inverse] Starting on %zu transactions. Max Support: %u, Min Abs Support: %u\n", 
           ds->count, max_sup, min_abs_sup);

    size_t total_sporadic = 0;
    size_t total_footprint = 0;

    IVLevel *levels = calloc(ds->max_id + 2, sizeof(IVLevel));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    // 1. Build Inverted Index (Vertical Level 1)
    uint32_t **temp_tids = calloc(ds->max_id + 1, sizeof(uint32_t *));
    size_t *temp_counts = calloc(ds->max_id + 1, sizeof(size_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) temp_counts[data[i].items[j]]++;
    }
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (temp_counts[i] > 0) temp_tids[i] = malloc(temp_counts[i] * sizeof(uint32_t));
    }
    size_t *curr_pos = calloc(ds->max_id + 1, sizeof(size_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            temp_tids[item][curr_pos[item]++] = (uint32_t)i;
        }
    }

    // Identify sporadic 1-itemsets
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (temp_counts[i] > 0) {
            if (temp_counts[i] < max_sup && temp_counts[i] >= min_abs_sup) {
                add_ivnode(&levels[1], &i, 1, temp_tids[i], temp_counts[i]);
                total_sporadic++;
                total_footprint++;
            }
        }
    }
    // Identify sporadic 1-itemsets
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (temp_counts[i] > 0) {
            if (temp_counts[i] < max_sup && temp_counts[i] >= min_abs_sup) {
                add_ivnode(&levels[1], &i, 1, temp_tids[i], temp_counts[i]);
                total_sporadic++;
                total_footprint++;
            }
        }
    }

    // Cleanup temp index
    for (uint32_t i = 0; i <= ds->max_id; i++) free(temp_tids[i]);
    free(temp_tids);
    free(temp_counts);
    free(curr_pos);

    // 2. Main Loop (k >= 2)
    for (size_t k = 2; levels[k-1].count > 0; k++) {
        for (size_t i = 0; i < levels[k-1].count; i++) {
            for (size_t j = i + 1; j < levels[k-1].count; j++) {
                bool joinable = true;
                for (size_t m = 0; m < k - 2; m++) {
                    if (levels[k-1].nodes[i].items[m] != levels[k-1].nodes[j].items[m]) {
                        joinable = false;
                        break;
                    }
                }
                if (joinable) {
                    uint32_t *cand_items = malloc(k * sizeof(uint32_t));
                    memcpy(cand_items, levels[k-1].nodes[i].items, (k-1) * sizeof(uint32_t));
                    cand_items[k-1] = levels[k-1].nodes[j].items[k-2];
                    
                    if (cand_items[k-1] < cand_items[k-2]) {
                        uint32_t tmp = cand_items[k-1];
                        cand_items[k-1] = cand_items[k-2];
                        cand_items[k-2] = tmp;
                    }

                    bool perfectly_sporadic_candidate = true;
                    uint32_t *subset = malloc((k-1) * sizeof(uint32_t));
                    for (size_t m = 0; m < k; m++) {
                        size_t s_idx = 0;
                        for (size_t n = 0; n < k; n++) if (m != n) subset[s_idx++] = cand_items[n];
                        
                        bool found = false;
                        for (size_t n = 0; n < levels[k-1].count; n++) {
                            if (memcmp(levels[k-1].nodes[n].items, subset, (k-1) * sizeof(uint32_t)) == 0) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) { perfectly_sporadic_candidate = false; break; }
                    }
                    free(subset);

                    if (perfectly_sporadic_candidate) {
                        uint32_t *new_tids = malloc(levels[k-1].nodes[i].tid_count * sizeof(uint32_t));
                        size_t new_count = intersect(levels[k-1].nodes[i].tids, levels[k-1].nodes[i].tid_count,
                                                   levels[k-1].nodes[j].tids, levels[k-1].nodes[j].tid_count,
                                                   new_tids);
                        
                        if (new_count >= min_abs_sup) {
                            add_ivnode(&levels[k], cand_items, k, new_tids, new_count);
                            total_sporadic++;
                            total_footprint += k;
                        }
                        free(new_tids);
                    }
                    free(cand_items);
                }
            }
        }
        if (levels[k].count == 0) break;
    }

    printf("[Apriori-Inverse] Complete. Total Perfectly Sporadic Itemsets: %zu\n", total_sporadic);
    dm_bench_record_results(total_sporadic, total_footprint);

    // Cleanup
    for (size_t i = 0; i <= ds->max_id + 1; i++) free_level(&levels[i]);
    free(levels);

    return DM_SUCCESS;
}

static DM_Algorithm algo_apriori_inverse = {
    .id = "apriori_inverse",
    .name = "Apriori-Inverse Algorithm",
    .description = "Vertical mining of perfectly sporadic itemsets (low support).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_apriori_inverse)
