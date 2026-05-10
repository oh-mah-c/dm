#include "algorithms/apriori.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

/* --- INTERNAL DATA STRUCTURES --- */

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t support;
} Apriori_Itemset;

typedef struct {
    Apriori_Itemset *array;
    size_t count;
    size_t capacity;
    size_t k;
} Apriori_List;

/* --- HASH TREE FOR FAST COUNTING --- */

#define HASH_BUCKETS 101
#define LEAF_THRESHOLD 10

typedef struct HashTreeNode {
    bool is_leaf;
    union {
        struct {
            Apriori_Itemset **itemsets;
            size_t count;
            size_t capacity;
        } leaf;
        struct {
            struct HashTreeNode **buckets;
        } interior;
    } data;
} HashTreeNode;

static HashTreeNode* create_leaf() {
    HashTreeNode *node = (HashTreeNode*)malloc(sizeof(HashTreeNode));
    node->is_leaf = true;
    node->data.leaf.count = 0;
    node->data.leaf.capacity = 4;
    node->data.leaf.itemsets = (Apriori_Itemset**)malloc(sizeof(Apriori_Itemset*) * node->data.leaf.capacity);
    return node;
}

static HashTreeNode* create_interior() {
    HashTreeNode *node = (HashTreeNode*)malloc(sizeof(HashTreeNode));
    node->is_leaf = false;
    node->data.interior.buckets = (HashTreeNode**)calloc(HASH_BUCKETS, sizeof(HashTreeNode*));
    return node;
}

static void hash_tree_free(HashTreeNode *node) {
    if (!node) return;
    if (node->is_leaf) {
        free(node->data.leaf.itemsets);
    } else {
        for (int i = 0; i < HASH_BUCKETS; i++) {
            hash_tree_free(node->data.interior.buckets[i]);
        }
        free(node->data.interior.buckets);
    }
    free(node);
}

static void hash_tree_insert(HashTreeNode *node, Apriori_Itemset *itemset, size_t depth) {
    if (node->is_leaf) {
        if (node->data.leaf.count < LEAF_THRESHOLD || depth >= itemset->count) {
            if (node->data.leaf.count >= node->data.leaf.capacity) {
                node->data.leaf.capacity *= 2;
                node->data.leaf.itemsets = (Apriori_Itemset**)realloc(node->data.leaf.itemsets, sizeof(Apriori_Itemset*) * node->data.leaf.capacity);
            }
            node->data.leaf.itemsets[node->data.leaf.count++] = itemset;
        } else {
            // Split leaf into interior
            Apriori_Itemset **old_itemsets = node->data.leaf.itemsets;
            size_t old_count = node->data.leaf.count;
            
            node->is_leaf = false;
            node->data.interior.buckets = (HashTreeNode**)calloc(HASH_BUCKETS, sizeof(HashTreeNode*));
            
            for (size_t i = 0; i < old_count; i++) {
                hash_tree_insert(node, old_itemsets[i], depth);
            }
            free(old_itemsets);
            hash_tree_insert(node, itemset, depth);
        }
    } else {
        uint32_t item = itemset->items[depth];
        uint32_t bucket_idx = item % HASH_BUCKETS;
        if (!node->data.interior.buckets[bucket_idx]) {
            node->data.interior.buckets[bucket_idx] = create_leaf();
        }
        hash_tree_insert(node->data.interior.buckets[bucket_idx], itemset, depth + 1);
    }
}

static void hash_tree_subset(HashTreeNode *node, uint32_t *t_items, size_t t_count, size_t t_start, size_t k, uint32_t *current_itemset, size_t current_len) {
    if (!node) return;
    if (node->is_leaf) {
        for (size_t i = 0; i < node->data.leaf.count; i++) {
            Apriori_Itemset *c = node->data.leaf.itemsets[i];
            // Check if c is a subset of the transaction.
            // Since we followed the tree, we only need to verify if the rest of c is in t.
            // However, a simple subset check on the full items is safer.
            bool match = true;
            size_t ti = 0;
            for (size_t ci = 0; ci < c->count; ci++) {
                bool found = false;
                while (ti < t_count) {
                    if (t_items[ti] == c->items[ci]) { found = true; ti++; break; }
                    if (t_items[ti] > c->items[ci]) break;
                    ti++;
                }
                if (!found) { match = false; break; }
            }
            if (match) c->support++;
        }
    } else {
        for (size_t i = t_start; i < t_count; i++) {
            uint32_t item = t_items[i];
            uint32_t bucket_idx = item % HASH_BUCKETS;
            hash_tree_subset(node->data.interior.buckets[bucket_idx], t_items, t_count, i + 1, k, current_itemset, current_len + 1);
        }
    }
}

/* --- LIST UTILS --- */

static void list_init(Apriori_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (Apriori_Itemset *)malloc(sizeof(Apriori_Itemset) * list->capacity);
}

static void list_free(Apriori_List *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->array[i].items);
    }
    free(list->array);
}

static void list_append(Apriori_List *list, const uint32_t *items, size_t count, uint32_t support) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (Apriori_Itemset *)realloc(list->array, sizeof(Apriori_Itemset) * list->capacity);
    }
    list->array[list->count].items = (uint32_t*)malloc(sizeof(uint32_t) * count);
    memcpy(list->array[list->count].items, items, sizeof(uint32_t) * count);
    list->array[list->count].count = count;
    list->array[list->count].support = support;
    list->count++;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static int cmp_itemset(const uint32_t *a, const uint32_t *b, size_t k) {
    for (size_t i = 0; i < k; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static bool list_contains(Apriori_List *list, const uint32_t *items) {
    size_t low = 0, high = list->count;
    size_t k = list->k;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = cmp_itemset(list->array[mid].items, items, k);
        if (cmp == 0) return true;
        if (cmp < 0) low = mid + 1;
        else high = mid;
    }
    return false;
}

/* --- APRIORI-GEN --- */
static void apriori_gen(Apriori_List *L_prev, Apriori_List *C_curr) {
    size_t k = L_prev->k + 1;
    list_init(C_curr, k);
    
    uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (k - 1));

    for (size_t i = 0; i < L_prev->count; i++) {
        for (size_t j = i + 1; j < L_prev->count; j++) {
            // Join step
            bool match = true;
            for (size_t m = 0; m < k - 2; m++) {
                if (L_prev->array[i].items[m] != L_prev->array[j].items[m]) {
                    match = false;
                    break;
                }
            }
            if (!match) break;

            uint32_t *c_items = (uint32_t *)malloc(sizeof(uint32_t) * k);
            memcpy(c_items, L_prev->array[i].items, sizeof(uint32_t) * (k - 1));
            c_items[k - 1] = L_prev->array[j].items[k - 2];
            // Items are already sorted lexicographically.

            // Prune step
            bool all_subsets_large = true;
            for (size_t drop = 0; drop < k; drop++) {
                if (drop == k - 1 || drop == k - 2) continue;
                size_t s_idx = 0;
                for (size_t m = 0; m < k; m++) {
                    if (m != drop) subset[s_idx++] = c_items[m];
                }
                if (!list_contains(L_prev, subset)) {
                    all_subsets_large = false;
                    break;
                }
            }

            if (all_subsets_large) {
                list_append(C_curr, c_items, k, 0);
            }
            free(c_items);
        }
    }
    free(subset);
}

/* --- RULE GENERATION --- */

static uint32_t get_support(Apriori_List *all_L, size_t total_levels, uint32_t *items, size_t k) {
    if (k == 0) return 0;
    Apriori_List *L = NULL;
    for (size_t i = 0; i < total_levels; i++) {
        if (all_L[i].k == k) { L = &all_L[i]; break; }
    }
    if (!L) return 0;
    
    size_t low = 0, high = L->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = cmp_itemset(L->array[mid].items, items, k);
        if (cmp == 0) return L->array[mid].support;
        if (cmp < 0) low = mid + 1;
        else high = mid;
    }
    return 0;
}

static void gen_rules(Apriori_Itemset l_k, Apriori_List *H_m, Apriori_List *all_L, size_t total_levels, double min_conf, size_t *rule_count) {
    size_t k = l_k.count;
    size_t m = H_m->k;
    
    if (k > m + 1) {
        Apriori_List H_next;
        apriori_gen(H_m, &H_next);
        
        for (size_t i = 0; i < H_next.count; ) {
            uint32_t *h = H_next.array[i].items;
            size_t ant_len = k - H_next.k;
            uint32_t *ant = (uint32_t*)malloc(sizeof(uint32_t) * ant_len);
            
            size_t ap = 0;
            for (size_t j = 0; j < k; j++) {
                bool in_h = false;
                for (size_t n = 0; n < H_next.k; n++) {
                    if (l_k.items[j] == h[n]) { in_h = true; break; }
                }
                if (!in_h) ant[ap++] = l_k.items[j];
            }
            
            uint32_t ant_supp = get_support(all_L, total_levels, ant, ant_len);
            free(ant);
            
            if (ant_supp > 0 && (double)l_k.support / ant_supp >= min_conf) {
                (*rule_count)++;
                i++;
            } else {
                // Delete from H_next
                free(H_next.array[i].items);
                for (size_t j = i; j < H_next.count - 1; j++) H_next.array[j] = H_next.array[j+1];
                H_next.count--;
            }
        }
        
        if (H_next.count > 0) {
            gen_rules(l_k, &H_next, all_L, total_levels, min_conf, rule_count);
        }
        list_free(&H_next);
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_APRIORI_Params *p = (DM_APRIORI_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    double min_conf = p ? p->min_confidence : 0.8;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[Apriori] Starting on %zu transactions. Min Support: %u, Min Confidence: %.2f\n", ds->count, min_sup, min_conf);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);
    }

    Apriori_List *all_L = (Apriori_List*)malloc(sizeof(Apriori_List) * 100);
    size_t levels = 0;

    // 1. Generate L1
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    list_init(&all_L[levels], 1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            uint32_t item = i;
            list_append(&all_L[levels], &item, 1, counts[i]);
        }
    }
    free(counts);
    printf("[Apriori] Found %zu 1-itemsets\n", all_L[levels].count);
    levels++;

    // 2. Iterative Pass
    while (all_L[levels - 1].count > 0) {
        Apriori_List C;
        apriori_gen(&all_L[levels - 1], &C);
        if (C.count == 0) { list_free(&C); break; }

        // Build Hash Tree
        HashTreeNode *root = create_leaf();
        for (size_t i = 0; i < C.count; i++) {
            hash_tree_insert(root, &C.array[i], 0);
        }

        // Count support using hash tree
        for (size_t i = 0; i < ds->count; i++) {
            if (data[i].count < C.k) continue;
            hash_tree_subset(root, data[i].items, data[i].count, 0, C.k, NULL, 0);
        }
        hash_tree_free(root);

        // Filter L
        list_init(&all_L[levels], C.k);
        for (size_t i = 0; i < C.count; i++) {
            if (C.array[i].support >= min_sup) {
                list_append(&all_L[levels], C.array[i].items, C.k, C.array[i].support);
            }
        }
        list_free(&C);
        
        printf("[Apriori] Found %zu %zu-itemsets\n", all_L[levels].count, all_L[levels].k);
        if (all_L[levels].count == 0) { list_free(&all_L[levels]); break; }
        levels++;
    }

    // 3. Rule Generation
    size_t rule_count = 0;
    printf("[Apriori] Phase 2: Generating rules...\n");
    for (size_t i = 1; i < levels; i++) {
        for (size_t j = 0; j < all_L[i].count; j++) {
            Apriori_Itemset l_k = all_L[i].array[j];
            
            // H1
            Apriori_List H1;
            list_init(&H1, 1);
            for (size_t m = 0; m < l_k.count; m++) {
                uint32_t item = l_k.items[m];
                list_append(&H1, &item, 1, 0);
            }
            
            for (size_t h_idx = 0; h_idx < H1.count; ) {
                uint32_t *h1 = H1.array[h_idx].items;
                size_t ant_len = l_k.count - 1;
                uint32_t *ant = (uint32_t*)malloc(sizeof(uint32_t) * ant_len);
                size_t ap = 0;
                for (size_t m = 0; m < l_k.count; m++) {
                    if (l_k.items[m] != h1[0]) ant[ap++] = l_k.items[m];
                }
                uint32_t ant_supp = get_support(all_L, levels, ant, ant_len);
                free(ant);
                
                if (ant_supp > 0 && (double)l_k.support / ant_supp >= min_conf) {
                    rule_count++;
                    h_idx++;
                } else {
                    free(H1.array[h_idx].items);
                    for (size_t m = h_idx; m < H1.count - 1; m++) H1.array[m] = H1.array[m+1];
                    H1.count--;
                }
            }
            
            if (H1.count > 0) {
                gen_rules(l_k, &H1, all_L, levels, min_conf, &rule_count);
            }
            list_free(&H1);
        }
    }
    printf("[Apriori] Phase 2 complete. Found %zu association rules.\n", rule_count);

    size_t total_fi = 0, footprint = 0;
    for (size_t i = 0; i < levels; i++) {
        total_fi += all_L[i].count;
        footprint += all_L[i].count * all_L[i].k;
        list_free(&all_L[i]);
    }
    free(all_L);

    printf("[Apriori] Complete. Total frequent itemsets: %zu\n", total_fi);
    dm_bench_record_results(total_fi, footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "apriori",
    .name = "Apriori Algorithm",
    .description = "Exact implementation from Agrawal & Srikant 1994 with Hash Tree and Rule Generation.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
