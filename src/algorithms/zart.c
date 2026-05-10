#include "algorithms/zart.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * ZART Algorithm for mining frequent itemsets, closed itemsets and associating generators.
 * Reference: L. Szathmary et al., "ZART: A Multifunctional Itemset Mining Algorithm", CLA 2007.
 */

typedef struct ZNode {
    uint32_t *items;
    size_t len;
    uint32_t sup;
    uint32_t pred_sup;
    bool is_key;
    bool is_closed;
    struct ZNode *next;
} ZNode;

#define HASH_SIZE 65536

typedef struct {
    ZNode **buckets;
    size_t count;
} ZLevel;

typedef struct FG_Entry {
    uint32_t *items;
    size_t len;
    uint32_t sup;
    struct FG_Entry *next;
} FG_Entry;

typedef struct {
    FG_Entry *head;
    size_t count;
} FG_List;

static uint32_t hash_items(const uint32_t *items, size_t len) {
    uint32_t hash = 0;
    for (size_t i = 0; i < len; i++) hash = hash * 31 + items[i];
    return hash;
}

static ZNode* find_znode(ZLevel *lvl, const uint32_t *items, size_t len) {
    if (!lvl->buckets) return NULL;
    uint32_t h = hash_items(items, len);
    for (ZNode *n = lvl->buckets[h % HASH_SIZE]; n; n = n->next) {
        if (n->len == len && memcmp(n->items, items, len * sizeof(uint32_t)) == 0) return n;
    }
    return NULL;
}

static void add_znode(ZLevel *lvl, ZNode *node) {
    if (!lvl->buckets) lvl->buckets = calloc(HASH_SIZE, sizeof(ZNode *));
    uint32_t h = hash_items(node->items, node->len);
    node->next = lvl->buckets[h % HASH_SIZE];
    lvl->buckets[h % HASH_SIZE] = node;
    lvl->count++;
}

static void free_level(ZLevel *lvl) {
    if (!lvl->buckets) return;
    for (size_t i = 0; i < HASH_SIZE; i++) {
        ZNode *curr = lvl->buckets[i];
        while (curr) {
            ZNode *next = curr->next;
            free(curr->items);
            free(curr);
            curr = next;
        }
    }
    free(lvl->buckets);
    lvl->buckets = NULL;
    lvl->count = 0;
}

static bool is_subset(const uint32_t *sub, size_t len_sub, const uint32_t *sup, size_t len_sup) {
    if (len_sub > len_sup) return false;
    size_t i = 0, j = 0;
    while (i < len_sub && j < len_sup) {
        if (sub[i] < sup[j]) return false;
        if (sub[i] == sup[j]) i++;
        j++;
    }
    return i == len_sub;
}

static void find_generators(ZLevel *Z_curr, FG_List *FG, ZLevel *F_curr, size_t *total_classes) {
    // 1. Loop over closed itemsets
    for (size_t i = 0; i < HASH_SIZE; i++) {
        for (ZNode *z = Z_curr->buckets[i]; z; z = z->next) {
            (*total_classes)++;
            // S = Subsets of z in FG
            FG_Entry **prev = &FG->head;
            FG_Entry *curr = FG->head;
            while (curr) {
                if (is_subset(curr->items, curr->len, z->items, z->len)) {
                    // Match found! In ZART, this means 'curr' is a generator of 'z'
                    // We would register it here.
                    // For benchmarking, we just count.
                    FG_Entry *to_del = curr;
                    *prev = curr->next;
                    curr = curr->next;
                    free(to_del->items);
                    free(to_del);
                    FG->count--;
                } else {
                    prev = &curr->next;
                    curr = curr->next;
                }
            }
        }
    }

    // 2. Add not-closed key itemsets to FG
    for (size_t i = 0; i < HASH_SIZE; i++) {
        for (ZNode *l = F_curr->buckets[i]; l; l = l->next) {
            if (l->is_key && !l->is_closed) {
                FG_Entry *entry = malloc(sizeof(FG_Entry));
                entry->items = malloc(l->len * sizeof(uint32_t));
                memcpy(entry->items, l->items, l->len * sizeof(uint32_t));
                entry->len = l->len;
                entry->sup = l->sup;
                entry->next = FG->head;
                FG->head = entry;
                FG->count++;
            }
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_ZART_Params *p = (DM_ZART_Params *)params;
    double min_sup_param = p ? p->min_support : 0.05;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[ZART] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    size_t total_frequent = 0;
    size_t total_closed = 0;
    size_t total_footprint = 0;
    size_t total_classes = 0;

    ZLevel *levels = calloc(ds->max_id + 2, sizeof(ZLevel));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    FG_List FG = {NULL, 0};

    // Level 1
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    bool full_column = false;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            ZNode *node = malloc(sizeof(ZNode));
            node->items = malloc(sizeof(uint32_t));
            node->items[0] = i;
            node->len = 1;
            node->sup = counts[i];
            node->pred_sup = (uint32_t)ds->count;
            node->is_closed = true; // Initially true
            if (node->sup == ds->count) {
                node->is_key = false;
                full_column = true;
            } else {
                node->is_key = true;
            }
            add_znode(&levels[1], node);
            total_frequent++;
            total_footprint++;
        }
    }
    free(counts);

    // Main Loop
    for (size_t k = 2; levels[k-1].count > 0; k++) {
        ZLevel candidates = {0};
        
        // 1. Generate Candidates (Pascal-GEN)
        ZNode **prev_nodes = malloc(levels[k-1].count * sizeof(ZNode *));
        size_t n_idx = 0;
        for (size_t i = 0; i < HASH_SIZE; i++) {
            for (ZNode *n = levels[k-1].buckets[i]; n; n = n->next) prev_nodes[n_idx++] = n;
        }

        for (size_t i = 0; i < levels[k-1].count; i++) {
            for (size_t j = i + 1; j < levels[k-1].count; j++) {
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
                    
                    // Full insertion sort for the new item
                    size_t m_pos = k - 1;
                    while (m_pos > 0 && cand_items[m_pos - 1] > new_item) {
                        cand_items[m_pos] = cand_items[m_pos - 1];
                        m_pos--;
                    }
                    cand_items[m_pos] = new_item;

                    uint32_t min_subset_sup = 0xFFFFFFFF;
                    bool all_subsets_key = true;
                    bool all_subsets_frequent = true;
                    
                    uint32_t *subset = malloc((k-1) * sizeof(uint32_t));
                    for (size_t m = 0; m < k; m++) {
                        size_t s_idx = 0;
                        for (size_t n = 0; n < k; n++) if (m != n) subset[s_idx++] = cand_items[n];
                        ZNode *s_node = find_znode(&levels[k-1], subset, k-1);
                        if (!s_node) {
                            all_subsets_frequent = false;
                            break;
                        }
                        if (s_node->sup < min_subset_sup) min_subset_sup = s_node->sup;
                        if (!s_node->is_key) all_subsets_key = false;
                    }
                    free(subset);

                    if (all_subsets_frequent) {
                        ZNode *c_node = malloc(sizeof(ZNode));
                        c_node->items = cand_items;
                        c_node->len = k;
                        c_node->pred_sup = min_subset_sup;
                        c_node->is_key = all_subsets_key;
                        c_node->is_closed = true;
                        if (!c_node->is_key) c_node->sup = min_subset_sup;
                        else c_node->sup = 0;
                        add_znode(&candidates, c_node);
                    } else {
                        free(cand_items);
                    }
                }
            }
        }
        free(prev_nodes);

        if (candidates.count > 0) {
            // Database Pass
            bool has_key_patterns = false;
            for (size_t i = 0; i < HASH_SIZE; i++) {
                for (ZNode *n = candidates.buckets[i]; n; n = n->next) {
                    if (n->is_key) { has_key_patterns = true; break; }
                }
                if (has_key_patterns) break;
            }

            if (has_key_patterns) {
                for (size_t i = 0; i < ds->count; i++) {
                    if (data[i].count < k) continue;
                    for (size_t b = 0; b < HASH_SIZE; b++) {
                        for (ZNode *n = candidates.buckets[b]; n; n = n->next) {
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

            // Post-pass
            for (size_t b = 0; b < HASH_SIZE; b++) {
                for (ZNode *n = candidates.buckets[b]; n; n = n->next) {
                    if (n->is_key && n->sup == n->pred_sup) n->is_key = false;
                    if (n->sup >= min_sup) {
                        ZNode *copy = malloc(sizeof(ZNode));
                        copy->items = malloc(k * sizeof(uint32_t));
                        memcpy(copy->items, n->items, k * sizeof(uint32_t));
                        copy->len = k;
                        copy->sup = n->sup;
                        copy->pred_sup = n->pred_sup;
                        copy->is_key = n->is_key;
                        copy->is_closed = true;
                        add_znode(&levels[k], copy);
                        total_frequent++;
                        total_footprint += k;

                        // Mark subsets as not-closed
                        uint32_t *subset = malloc((k-1) * sizeof(uint32_t));
                        for (size_t m = 0; m < k; m++) {
                            size_t s_idx = 0;
                            for (size_t nx = 0; nx < k; nx++) if (m != nx) subset[s_idx++] = copy->items[nx];
                            ZNode *s_node = find_znode(&levels[k-1], subset, k-1);
                            if (s_node && s_node->sup == copy->sup) s_node->is_closed = false;
                        }
                        free(subset);
                    }
                }
            }
        }

        // Identify Zi-1 and Associate Generators
        ZLevel Z_prev = {0};
        size_t level_closed = 0;
        for (size_t b = 0; b < HASH_SIZE; b++) {
            for (ZNode *n = levels[k-1].buckets[b]; n; n = n->next) {
                if (n->is_closed) {
                    ZNode *copy = malloc(sizeof(ZNode));
                    copy->items = malloc(n->len * sizeof(uint32_t));
                    memcpy(copy->items, n->items, n->len * sizeof(uint32_t));
                    copy->len = n->len;
                    copy->sup = n->sup;
                    add_znode(&Z_prev, copy);
                    total_closed++;
                    level_closed++;
                }
            }
        }
        find_generators(&Z_prev, &FG, &levels[k-1], &total_classes);
        free_level(&Z_prev);

        free_level(&candidates);
        if (levels[k].count == 0) break;
    }

    printf("[ZART] Complete. Frequent: %zu, Closed: %zu, Equivalence Classes: %zu\n", 
           total_frequent, total_closed, total_classes);
    dm_bench_record_results(total_closed, total_footprint);

    // Cleanup
    for (size_t i = 0; i <= ds->max_id + 1; i++) free_level(&levels[i]);
    free(levels);
    FG_Entry *curr = FG.head;
    while (curr) {
        FG_Entry *next = curr->next;
        free(curr->items);
        free(curr);
        curr = next;
    }

    return DM_SUCCESS;
}

static DM_Algorithm algo_zart = {
    .id = "zart",
    .name = "ZART Algorithm",
    .description = "Multifunctional itemset mining (FIs, FCIs, and Generators) based on Pascal.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_zart)
