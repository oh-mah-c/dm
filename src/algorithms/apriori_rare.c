#include "algorithms/apriori_rare.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Apriori-Rare Algorithm for mining minimal rare itemsets (mRIs).
 * Reference: L. Szathmary et al., "Towards Rare Itemset Mining", ICTAI 2007.
 */

typedef struct RNode {
    uint32_t *items;
    size_t len;
    uint32_t sup;
    struct RNode *next;
} RNode;

#define HASH_SIZE 65536

typedef struct {
    RNode **buckets;
    size_t count;
} RLevel;

static uint32_t hash_items(const uint32_t *items, size_t len) {
    uint32_t hash = 0;
    for (size_t i = 0; i < len; i++) hash = hash * 31 + items[i];
    return hash;
}

static RNode* find_rnode(RLevel *lvl, const uint32_t *items, size_t len) {
    if (!lvl->buckets) return NULL;
    uint32_t h = hash_items(items, len);
    for (RNode *n = lvl->buckets[h % HASH_SIZE]; n; n = n->next) {
        if (n->len == len && memcmp(n->items, items, len * sizeof(uint32_t)) == 0) return n;
    }
    return NULL;
}

static void add_rnode(RLevel *lvl, RNode *node) {
    if (!lvl->buckets) lvl->buckets = calloc(HASH_SIZE, sizeof(RNode *));
    uint32_t h = hash_items(node->items, node->len);
    node->next = lvl->buckets[h % HASH_SIZE];
    lvl->buckets[h % HASH_SIZE] = node;
    lvl->count++;
}

static void free_level(RLevel *lvl) {
    if (!lvl->buckets) return;
    for (size_t i = 0; i < HASH_SIZE; i++) {
        RNode *curr = lvl->buckets[i];
        while (curr) {
            RNode *next = curr->next;
            free(curr->items);
            free(curr);
            curr = next;
        }
    }
    free(lvl->buckets);
    lvl->buckets = NULL;
    lvl->count = 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_APRIORI_RARE_Params *p = (DM_APRIORI_RARE_Params *)params;
    double min_sup_param = p ? p->min_support : 0.05;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[Apriori-Rare] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    size_t total_mris = 0;
    size_t total_frequent = 0;
    size_t total_footprint = 0;

    RLevel *f_levels = calloc(ds->max_id + 2, sizeof(RLevel));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    // Level 1
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] > 0) {
            RNode *node = malloc(sizeof(RNode));
            node->items = malloc(sizeof(uint32_t));
            node->items[0] = i;
            node->len = 1;
            node->sup = counts[i];
            
            if (node->sup < min_sup) {
                // Minimal Rare Itemset (Level 1 rare is always minimal)
                total_mris++;
                // In a real app, we would save this to a results list
                free(node->items);
                free(node);
            } else {
                add_rnode(&f_levels[1], node);
                total_frequent++;
            }
        }
    }
    free(counts);

    // Main Loop
    for (size_t k = 2; f_levels[k-1].count > 0; k++) {
        RLevel candidates = {0};
        
        // 1. Generate Candidates
        RNode **prev_nodes = malloc(f_levels[k-1].count * sizeof(RNode *));
        size_t n_idx = 0;
        for (size_t i = 0; i < HASH_SIZE; i++) {
            for (RNode *n = f_levels[k-1].buckets[i]; n; n = n->next) prev_nodes[n_idx++] = n;
        }

        for (size_t i = 0; i < f_levels[k-1].count; i++) {
            for (size_t j = i + 1; j < f_levels[k-1].count; j++) {
                bool joinable = true;
                for (size_t m = 0; m < k - 2; m++) {
                    if (prev_nodes[i]->items[m] != prev_nodes[j]->items[m]) {
                        joinable = false;
                        break;
                    }
                }
                if (joinable) {
                    uint32_t *cand_items = malloc(k * sizeof(uint32_t));
                    memcpy(cand_items, prev_nodes[i]->items, (k-1) * sizeof(uint32_t));
                    uint32_t new_item = prev_nodes[j]->items[k-2];
                    
                    // Sorted insertion
                    size_t m_pos = k - 1;
                    while (m_pos > 0 && cand_items[m_pos - 1] > new_item) {
                        cand_items[m_pos] = cand_items[m_pos - 1];
                        m_pos--;
                    }
                    cand_items[m_pos] = new_item;

                    // Prune: all subsets must be frequent
                    bool all_subsets_frequent = true;
                    uint32_t *subset = malloc((k-1) * sizeof(uint32_t));
                    for (size_t m = 0; m < k; m++) {
                        size_t s_idx = 0;
                        for (size_t n = 0; n < k; n++) if (m != n) subset[s_idx++] = cand_items[n];
                        if (!find_rnode(&f_levels[k-1], subset, k-1)) {
                            all_subsets_frequent = false;
                            break;
                        }
                    }
                    free(subset);

                    if (all_subsets_frequent) {
                        RNode *c_node = malloc(sizeof(RNode));
                        c_node->items = cand_items;
                        c_node->len = k;
                        c_node->sup = 0;
                        add_rnode(&candidates, c_node);
                    } else {
                        free(cand_items);
                    }
                }
            }
        }
        free(prev_nodes);

        if (candidates.count == 0) {
            free_level(&candidates);
            break;
        }

        // 2. Database Pass
        for (size_t i = 0; i < ds->count; i++) {
            if (data[i].count < k) continue;
            for (size_t b = 0; b < HASH_SIZE; b++) {
                for (RNode *n = candidates.buckets[b]; n; n = n->next) {
                    bool match = true;
                    for (size_t m = 0; m < k; m++) {
                        bool found = false;
                        for (size_t tx = 0; tx < data[i].count; tx++) {
                            if (data[i].items[tx] == n->items[m]) { found = true; break; }
                        }
                        if (!found) { match = false; break; }
                    }
                    if (match) n->sup++;
                }
            }
        }

        // 3. Collect Frequent and mRIs
        for (size_t b = 0; b < HASH_SIZE; b++) {
            RNode *n = candidates.buckets[b];
            while (n) {
                if (n->sup < min_sup) {
                    total_mris++;
                } else {
                    RNode *copy = malloc(sizeof(RNode));
                    copy->items = malloc(k * sizeof(uint32_t));
                    memcpy(copy->items, n->items, k * sizeof(uint32_t));
                    copy->len = k;
                    copy->sup = n->sup;
                    add_rnode(&f_levels[k], copy);
                    total_frequent++;
                    total_footprint += k;
                }
                n = n->next;
            }
        }
        free_level(&candidates);
        if (f_levels[k].count == 0) break;
    }

    printf("[Apriori-Rare] Complete. Minimal Rare Itemsets (mRIs): %zu\n", total_mris);
    printf("[Apriori-Rare] Frequent Itemsets found during traversal: %zu\n", total_frequent);
    dm_bench_record_results(total_mris, total_footprint);

    // Cleanup
    for (size_t i = 0; i <= ds->max_id + 1; i++) free_level(&f_levels[i]);
    free(f_levels);

    return DM_SUCCESS;
}

static DM_Algorithm algo_apriori_rare = {
    .id = "apriori_rare",
    .name = "Apriori-Rare Algorithm",
    .description = "Level-wise mining of minimal rare itemsets (mRIs).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_apriori_rare)
