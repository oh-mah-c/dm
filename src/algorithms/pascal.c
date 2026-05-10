#include "algorithms/pascal.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * PASCAL Algorithm for mining frequent patterns using counting inference.
 * Reference: N. Pasquier et al., "Mining Frequent Patterns with Counting Inference", SIGKDD Explorations, 2000.
 */

typedef struct PNode {
    uint32_t *items;
    size_t len;
    uint32_t sup;
    uint32_t pred_sup;
    bool is_key;
    struct PNode *next;
} PNode;

#define HASH_SIZE 65536

typedef struct {
    PNode **buckets;
    size_t count;
} PLevel;

static uint32_t hash_items(const uint32_t *items, size_t len) {
    uint32_t hash = 0;
    for (size_t i = 0; i < len; i++) hash = hash * 31 + items[i];
    return hash;
}

static PNode* find_node(PLevel *lvl, const uint32_t *items, size_t len) {
    if (!lvl->buckets) return NULL;
    uint32_t h = hash_items(items, len);
    for (PNode *n = lvl->buckets[h % HASH_SIZE]; n; n = n->next) {
        if (n->len == len && memcmp(n->items, items, len * sizeof(uint32_t)) == 0) return n;
    }
    return NULL;
}

static void add_node(PLevel *lvl, PNode *node) {
    if (!lvl->buckets) lvl->buckets = calloc(HASH_SIZE, sizeof(PNode *));
    uint32_t h = hash_items(node->items, node->len);
    node->next = lvl->buckets[h % HASH_SIZE];
    lvl->buckets[h % HASH_SIZE] = node;
    lvl->count++;
}

static void free_level(PLevel *lvl) {
    if (!lvl->buckets) return;
    for (size_t i = 0; i < HASH_SIZE; i++) {
        PNode *curr = lvl->buckets[i];
        while (curr) {
            PNode *next = curr->next;
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
    DM_PASCAL_Params *p = (DM_PASCAL_Params *)params;
    double min_sup_param = p ? p->min_support : 0.05;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[PASCAL] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    size_t total_frequent = 0;
    size_t total_footprint = 0;
    size_t total_db_counts = 0;
    size_t total_inferred = 0;

    PLevel *levels = calloc(ds->max_id + 2, sizeof(PLevel));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    // Level 1
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            PNode *node = malloc(sizeof(PNode));
            node->items = malloc(sizeof(uint32_t));
            node->items[0] = i;
            node->len = 1;
            node->sup = counts[i];
            node->pred_sup = (uint32_t)ds->count; // Root support
            node->is_key = (node->sup != node->pred_sup);
            add_node(&levels[1], node);
            total_frequent++;
            total_footprint++;
            total_db_counts++;
        }
    }
    free(counts);

    // Main Loop
    for (size_t k = 2; levels[k-1].count > 0; k++) {
        PLevel candidates = {0};
        
        // 1. Generate Candidates (Apriori-GEN style)
        // Extract L_{k-1} nodes into a temporary array for faster iteration
        PNode **nodes = malloc(levels[k-1].count * sizeof(PNode *));
        size_t n_idx = 0;
        for (size_t i = 0; i < HASH_SIZE; i++) {
            for (PNode *n = levels[k-1].buckets[i]; n; n = n->next) nodes[n_idx++] = n;
        }

        for (size_t i = 0; i < levels[k-1].count; i++) {
            for (size_t j = i + 1; j < levels[k-1].count; j++) {
                // Join if first k-2 items are the same
                bool joinable = true;
                for (size_t m = 0; m < k - 2; m++) {
                    if (nodes[i]->items[m] != nodes[j]->items[m]) {
                        joinable = false;
                        break;
                    }
                }
                if (joinable) {
                    uint32_t *cand_items = malloc(k * sizeof(uint32_t));
                    memcpy(cand_items, nodes[i]->items, (k-1) * sizeof(uint32_t));
                    cand_items[k-1] = nodes[j]->items[k-2];
                    // Sort the last two items
                    if (cand_items[k-1] < cand_items[k-2]) {
                        uint32_t tmp = cand_items[k-1];
                        cand_items[k-1] = cand_items[k-2];
                        cand_items[k-2] = tmp;
                    }

                    // Prune and Inference Logic
                    uint32_t min_subset_sup = 0xFFFFFFFF;
                    bool all_subsets_key = true;
                    bool all_subsets_frequent = true;
                    
                    uint32_t *subset = malloc((k-1) * sizeof(uint32_t));
                    for (size_t m = 0; m < k; m++) {
                        size_t s_idx = 0;
                        for (size_t n = 0; n < k; n++) if (m != n) subset[s_idx++] = cand_items[n];
                        PNode *s_node = find_node(&levels[k-1], subset, k-1);
                        if (!s_node) {
                            all_subsets_frequent = false;
                            break;
                        }
                        if (s_node->sup < min_subset_sup) min_subset_sup = s_node->sup;
                        if (!s_node->is_key) all_subsets_key = false;
                    }
                    free(subset);

                    if (all_subsets_frequent) {
                        PNode *c_node = malloc(sizeof(PNode));
                        c_node->items = cand_items;
                        c_node->len = k;
                        c_node->pred_sup = min_subset_sup;
                        c_node->is_key = all_subsets_key;
                        if (!c_node->is_key) {
                            c_node->sup = min_subset_sup; // Inferred!
                            total_inferred++;
                        } else {
                            c_node->sup = 0; // To be counted
                        }
                        add_node(&candidates, c_node);
                    } else {
                        free(cand_items);
                    }
                }
            }
        }
        free(nodes);

        if (candidates.count == 0) {
            free_level(&candidates);
            break;
        }

        // 2. Database Pass (only for key patterns)
        bool has_key_patterns = false;
        for (size_t i = 0; i < HASH_SIZE; i++) {
            for (PNode *n = candidates.buckets[i]; n; n = n->next) {
                if (n->is_key) { has_key_patterns = true; break; }
            }
            if (has_key_patterns) break;
        }

        if (has_key_patterns) {
            for (size_t i = 0; i < ds->count; i++) {
                // Optimization: skip transaction if it doesn't contain enough items
                if (data[i].count < k) continue;
                
                // Simple subset check for all key candidates
                // In a production algorithm, a hash-tree or trie would be used here.
                for (size_t b = 0; b < HASH_SIZE; b++) {
                    for (PNode *n = candidates.buckets[b]; n; n = n->next) {
                        if (n->is_key) {
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
            }
        }

        // 3. Post-pass Inference Check and Pruning
        for (size_t b = 0; b < HASH_SIZE; b++) {
            PNode *n = candidates.buckets[b];
            while (n) {
                if (n->is_key) {
                    total_db_counts++;
                    // After counting, check if it's actually a key pattern
                    if (n->sup == n->pred_sup) n->is_key = false;
                }

                if (n->sup >= min_sup) {
                    PNode *copy = malloc(sizeof(PNode));
                    copy->items = malloc(k * sizeof(uint32_t));
                    memcpy(copy->items, n->items, k * sizeof(uint32_t));
                    copy->len = k;
                    copy->sup = n->sup;
                    copy->pred_sup = n->pred_sup;
                    copy->is_key = n->is_key;
                    add_node(&levels[k], copy);
                    total_frequent++;
                    total_footprint += k;
                }
                n = n->next;
            }
        }
        free_level(&candidates);
        if (levels[k].count == 0) break;
    }

    printf("[PASCAL] Complete. Frequent patterns: %zu, Footprint: %zu\n", total_frequent, total_footprint);
    printf("[PASCAL] Counting Inference: %zu patterns inferred, %zu patterns counted in DB.\n", total_inferred, total_db_counts);
    dm_bench_record_results(total_frequent, total_footprint);

    // Cleanup
    for (size_t i = 0; i <= ds->max_id + 1; i++) free_level(&levels[i]);
    free(levels);

    return DM_SUCCESS;
}

static DM_Algorithm algo_pascal = {
    .id = "pascal",
    .name = "PASCAL Algorithm",
    .description = "Level-wise frequent pattern mining using pattern counting inference.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_pascal)
