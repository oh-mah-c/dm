#include "algorithms/rp_tree.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * RP-Tree Algorithm for mining rare-item itemsets.
 * Reference: S. Tsang, Y. S. Koh, and G. Dobbie, "RP-Tree: Rare Pattern Tree Mining", DaWaK 2011.
 */

typedef struct RPNode {
    uint32_t item;
    uint32_t count;
    struct RPNode *parent;
    struct RPNode *next_homonym;
    struct RPNode **children;
    uint32_t num_children;
    uint32_t cap_children;
} RPNode;

typedef struct {
    uint32_t item;
    uint32_t support;
    RPNode *head;
} RPHeader;

typedef struct {
    RPHeader *headers;
    size_t count;
    int *item_to_idx;
} RPHeaderTable;

typedef struct {
    uint32_t min_freq_sup;
    uint32_t min_rare_sup;
    size_t total_rare_itemsets;
    size_t total_footprint;
    bool *is_rare;
} RPContext;

static RPNode* create_node(uint32_t item, uint32_t count, RPNode *parent) {
    RPNode *node = calloc(1, sizeof(RPNode));
    node->item = item;
    node->count = count;
    node->parent = parent;
    return node;
}

static void free_tree(RPNode *node) {
    if (!node) return;
    for (size_t i = 0; i < node->num_children; i++) free_tree(node->children[i]);
    free(node->children);
    free(node);
}

static void insert_tree(RPNode *root, uint32_t *items, size_t len, uint32_t count, RPHeaderTable *ht) {
    RPNode *curr = root;
    for (size_t i = 0; i < len; i++) {
        uint32_t item = items[i];
        RPNode *child = NULL;
        
        // Use binary search for children
        int low = 0, high = (int)curr->num_children - 1;
        int insert_pos = 0;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (curr->children[mid]->item == item) {
                child = curr->children[mid];
                break;
            }
            if (curr->children[mid]->item < item) {
                low = mid + 1;
                insert_pos = low;
            } else {
                high = mid - 1;
                insert_pos = mid;
            }
        }

        if (child) {
            child->count += count;
        } else {
            child = create_node(item, count, curr);
            if (curr->num_children >= curr->cap_children) {
                curr->cap_children = (curr->cap_children == 0) ? 4 : curr->cap_children * 2;
                curr->children = realloc(curr->children, curr->cap_children * sizeof(RPNode *));
            }
            if (insert_pos < (int)curr->num_children) {
                memmove(&curr->children[insert_pos + 1], &curr->children[insert_pos], (curr->num_children - insert_pos) * sizeof(RPNode *));
            }
            curr->children[insert_pos] = child;
            curr->num_children++;
            
            int idx = ht->item_to_idx[item];
            child->next_homonym = ht->headers[idx].head;
            ht->headers[idx].head = child;
        }
        curr = child;
    }
}

static int compare_items_desc(void *arg, const void *a, const void *b) {
    uint32_t *counts = (uint32_t *)arg;
    uint32_t i1 = *(uint32_t *)a;
    uint32_t i2 = *(uint32_t *)b;
    if (counts[i1] != counts[i2]) return (int)counts[i2] - (int)counts[i1];
    return (int)i1 - (int)i2;
}

static void mine_recursive(RPContext *ctx, RPNode *root, RPHeaderTable *ht, uint32_t *suffix, size_t suffix_len, bool has_rare_item) {
    // Standard FP-Growth recursive mining
    for (int i = (int)ht->count - 1; i >= 0; i--) {
        uint32_t item = ht->headers[i].item;

        // RP-Tree Optimization: If we haven't encountered a rare item yet, 
        // and this item is not rare, skip it. Frequent items never have rare ancestors.
        if (!has_rare_item && !ctx->is_rare[item]) continue;

        uint32_t support = 0;
        RPNode *curr = ht->headers[i].head;
        while (curr) { support += curr->count; curr = curr->next_homonym; }

        if (support >= ctx->min_rare_sup) {
            uint32_t *new_suffix = malloc((suffix_len + 1) * sizeof(uint32_t));
            new_suffix[0] = item;
            memcpy(new_suffix + 1, suffix, suffix_len * sizeof(uint32_t));
            
            bool new_has_rare = has_rare_item || ctx->is_rare[item];
            
            if (new_has_rare && support < ctx->min_freq_sup) {
                ctx->total_rare_itemsets++;
                ctx->total_footprint += suffix_len + 1;
            }

            // Build conditional pattern base
            typedef struct { uint32_t *items; size_t len; uint32_t count; } Path;
            size_t num_paths = 0;
            curr = ht->headers[i].head;
            while (curr) { num_paths++; curr = curr->next_homonym; }

            if (num_paths > 0) {
                Path *paths = malloc(num_paths * sizeof(Path));
                size_t p_idx = 0;
                curr = ht->headers[i].head;
                while (curr) {
                    size_t p_len = 0;
                    RPNode *p = curr->parent;
                    while (p && p->item != 0xFFFFFFFF) { p_len++; p = p->parent; }
                    
                    paths[p_idx].items = malloc(p_len * sizeof(uint32_t));
                    paths[p_idx].len = p_len;
                    paths[p_idx].count = curr->count;
                    
                    p = curr->parent;
                    for (int j = (int)p_len - 1; j >= 0; j--) {
                        paths[p_idx].items[j] = p->item;
                        p = p->parent;
                    }
                    p_idx++;
                    curr = curr->next_homonym;
                }

                // Build conditional tree
                uint32_t *cond_counts = calloc(ht->item_to_idx[item] + 1000, sizeof(uint32_t)); // Rough estimate for max_id
                uint32_t max_cond_id = 0;
                for (size_t j = 0; j < num_paths; j++) {
                    for (size_t m = 0; m < paths[j].len; m++) {
                        cond_counts[paths[j].items[m]] += paths[j].count;
                        if (paths[j].items[m] > max_cond_id) max_cond_id = paths[j].items[m];
                    }
                }

                RPHeaderTable cond_ht = {0};
                cond_ht.item_to_idx = malloc((max_cond_id + 1) * sizeof(int));
                for (uint32_t j = 0; j <= max_cond_id; j++) cond_ht.item_to_idx[j] = -1;

                for (size_t j = 0; j < i; j++) {
                    uint32_t it = ht->headers[j].item;
                    if (it <= max_cond_id && cond_counts[it] >= ctx->min_rare_sup) {
                        cond_ht.headers = realloc(cond_ht.headers, (cond_ht.count + 1) * sizeof(RPHeader));
                        cond_ht.headers[cond_ht.count].item = it;
                        cond_ht.headers[cond_ht.count].support = cond_counts[it];
                        cond_ht.headers[cond_ht.count].head = NULL;
                        cond_ht.item_to_idx[it] = (int)cond_ht.count;
                        cond_ht.count++;
                    }
                }

                if (cond_ht.count > 0) {
                    RPNode *cond_root = create_node(0xFFFFFFFF, 0, NULL);
                    for (size_t j = 0; j < num_paths; j++) {
                        uint32_t *filtered = malloc(paths[j].len * sizeof(uint32_t));
                        size_t f_len = 0;
                        for (size_t m = 0; m < paths[j].len; m++) {
                            if (cond_ht.item_to_idx[paths[j].items[m]] != -1) {
                                filtered[f_len++] = paths[j].items[m];
                            }
                        }
                        if (f_len > 0) {
                            // Order matches the header table order (descending support)
                            insert_tree(cond_root, filtered, f_len, paths[j].count, &cond_ht);
                        }
                        free(filtered);
                    }
                    mine_recursive(ctx, cond_root, &cond_ht, new_suffix, suffix_len + 1, new_has_rare);
                    free_tree(cond_root);
                }

                for (size_t j = 0; j < num_paths; j++) free(paths[j].items);
                free(paths);
                free(cond_ht.headers);
                free(cond_ht.item_to_idx);
                free(cond_counts);
            }
            free(new_suffix);
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_RP_TREE_Params *p = (DM_RP_TREE_Params *)params;
    double min_freq_param = p ? p->min_freq_support : 0.15;
    double min_rare_param = p ? p->min_rare_support : 0.05;
    
    uint32_t min_freq_sup = (min_freq_param < 1.0) ? (uint32_t)ceil(min_freq_param * ds->count) : (uint32_t)min_freq_param;
    uint32_t min_rare_sup = (min_rare_param < 1.0) ? (uint32_t)ceil(min_rare_param * ds->count) : (uint32_t)min_rare_param;
    if (min_rare_sup == 0 && ds->count > 0) min_rare_sup = 1;

    printf("[RP-Tree] Starting. Min Freq: %u, Min Rare: %u\n", min_freq_sup, min_rare_sup);

    RPContext ctx = {
        .min_freq_sup = min_freq_sup,
        .min_rare_sup = min_rare_sup,
        .total_rare_itemsets = 0,
        .total_footprint = 0,
        .is_rare = calloc(ds->max_id + 1, sizeof(bool))
    };

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    RPHeaderTable ht = {0};
    ht.item_to_idx = malloc((ds->max_id + 1) * sizeof(int));
    for (uint32_t i = 0; i <= ds->max_id; i++) ht.item_to_idx[i] = -1;

    uint32_t *sorted_items = malloc((ds->max_id + 1) * sizeof(uint32_t));
    size_t num_sorted = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_rare_sup) {
            sorted_items[num_sorted++] = i;
            if (counts[i] < min_freq_sup) ctx.is_rare[i] = true;
        }
    }
    qsort_s(sorted_items, num_sorted, sizeof(uint32_t), compare_items_desc, counts);

    for (size_t i = 0; i < num_sorted; i++) {
        ht.headers = realloc(ht.headers, (ht.count + 1) * sizeof(RPHeader));
        ht.headers[ht.count].item = sorted_items[i];
        ht.headers[ht.count].support = counts[sorted_items[i]];
        ht.headers[ht.count].head = NULL;
        ht.item_to_idx[sorted_items[i]] = (int)ht.count;
        ht.count++;
    }

    RPNode *root = create_node(0xFFFFFFFF, 0, NULL);
    for (size_t i = 0; i < ds->count; i++) {
        bool has_rare = false;
        for (size_t j = 0; j < data[i].count; j++) {
            if (ctx.is_rare[data[i].items[j]]) { has_rare = true; break; }
        }
        
        if (has_rare) {
            uint32_t *filtered = malloc(data[i].count * sizeof(uint32_t));
            size_t f_len = 0;
            for (size_t j = 0; j < data[i].count; j++) {
                if (ht.item_to_idx[data[i].items[j]] != -1) {
                    filtered[f_len++] = data[i].items[j];
                }
            }
            if (f_len > 0) {
                qsort_s(filtered, f_len, sizeof(uint32_t), compare_items_desc, counts);
                insert_tree(root, filtered, f_len, 1, &ht);
            }
            free(filtered);
        }
    }

    // Mining only starts from rare items as specified in RP-Tree Algorithm 1
    // But standard FP-Growth loop handles suffix selection.
    // To implement "only for rare items", we filter the first level of items.
    
    // We can just call mine_recursive and it will count only itemsets with at least one rare item.
    mine_recursive(&ctx, root, &ht, NULL, 0, false);

    printf("[RP-Tree] Complete. Total Rare-item Itemsets: %zu\n", ctx.total_rare_itemsets);
    dm_bench_record_results(ctx.total_rare_itemsets, ctx.total_footprint);

    free_tree(root);
    free(ht.headers);
    free(ht.item_to_idx);
    free(counts);
    free(sorted_items);
    free(ctx.is_rare);

    return DM_SUCCESS;
}

static DM_Algorithm algo_rp_tree = {
    .id = "rp_tree",
    .name = "RP-Tree Algorithm",
    .description = "Tree-based mining of rare-item itemsets.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_rp_tree)
