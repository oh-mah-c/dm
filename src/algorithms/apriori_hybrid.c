#include "algorithms/apriori_hybrid.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

/* --- INTERNAL DATA STRUCTURES (SHARED) --- */

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

typedef struct {
    uint32_t tid;
    uint32_t *ids;
    size_t count;
    size_t capacity;
} C_Prime_Entry;

typedef struct {
    C_Prime_Entry *entries;
    size_t count;
    size_t capacity;
} C_Prime_List;

typedef struct {
    uint32_t q_id;
    uint32_t c_id;
} JoinResult;

typedef struct {
    JoinResult *results;
    size_t count;
    size_t capacity;
} JoinMapEntry;

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

/* --- SHARED UTILS --- */

static void list_init(Apriori_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (Apriori_Itemset *)malloc(sizeof(Apriori_Itemset) * list->capacity);
}

static void list_free(Apriori_List *list) {
    for (size_t i = 0; i < list->count; i++) free(list->array[i].items);
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

static bool is_subset(const uint32_t *sub, size_t sub_len, const uint32_t *set, size_t set_len) {
    size_t i = 0, j = 0;
    while (i < sub_len && j < set_len) {
        if (sub[i] == set[j]) { i++; j++; }
        else if (sub[i] > set[j]) j++;
        else return false;
    }
    return i == sub_len;
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

/* --- HASH TREE UTILS --- */

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
    if (node->is_leaf) free(node->data.leaf.itemsets);
    else {
        for (int i = 0; i < HASH_BUCKETS; i++) hash_tree_free(node->data.interior.buckets[i]);
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
            Apriori_Itemset **old_itemsets = node->data.leaf.itemsets;
            size_t old_count = node->data.leaf.count;
            node->is_leaf = false;
            node->data.interior.buckets = (HashTreeNode**)calloc(HASH_BUCKETS, sizeof(HashTreeNode*));
            for (size_t i = 0; i < old_count; i++) hash_tree_insert(node, old_itemsets[i], depth);
            free(old_itemsets);
            hash_tree_insert(node, itemset, depth);
        }
    } else {
        uint32_t bucket_idx = itemset->items[depth] % HASH_BUCKETS;
        if (!node->data.interior.buckets[bucket_idx]) node->data.interior.buckets[bucket_idx] = create_leaf();
        hash_tree_insert(node->data.interior.buckets[bucket_idx], itemset, depth + 1);
    }
}

static void hash_tree_subset(HashTreeNode *node, uint32_t *t_items, size_t t_count, size_t t_start, size_t k) {
    if (!node) return;
    if (node->is_leaf) {
        for (size_t i = 0; i < node->data.leaf.count; i++) {
            Apriori_Itemset *c = node->data.leaf.itemsets[i];
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
            hash_tree_subset(node->data.interior.buckets[t_items[i] % HASH_BUCKETS], t_items, t_count, i + 1, k);
        }
    }
}

/* --- APRIORI-GEN (WITH JOIN MAP SUPPORT) --- */
static void apriori_gen_hybrid(Apriori_List *L_prev, Apriori_List *C_curr, JoinMapEntry *join_map) {
    size_t k = L_prev->k + 1;
    list_init(C_curr, k);
    uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (k - 1));
    for (size_t i = 0; i < L_prev->count; i++) {
        for (size_t j = i + 1; j < L_prev->count; j++) {
            bool match = true;
            for (size_t m = 0; m < k - 2; m++) if (L_prev->array[i].items[m] != L_prev->array[j].items[m]) { match = false; break; }
            if (!match) break;
            uint32_t *c_items = (uint32_t *)malloc(sizeof(uint32_t) * k);
            memcpy(c_items, L_prev->array[i].items, sizeof(uint32_t) * (k - 1));
            c_items[k - 1] = L_prev->array[j].items[k - 2];
            bool all_subsets_large = true;
            for (size_t drop = 0; drop < k; drop++) {
                if (drop == k - 1 || drop == k - 2) continue;
                size_t s_idx = 0;
                for (size_t m = 0; m < k; m++) if (m != drop) subset[s_idx++] = c_items[m];
                if (!list_contains(L_prev, subset)) { all_subsets_large = false; break; }
            }
            if (all_subsets_large) {
                uint32_t c_id = C_curr->count;
                list_append(C_curr, c_items, k, 0);
                if (join_map) {
                    JoinMapEntry *entry = &join_map[i];
                    if (entry->count >= entry->capacity) {
                        entry->capacity = entry->capacity == 0 ? 4 : entry->capacity * 2;
                        entry->results = (JoinResult*)realloc(entry->results, sizeof(JoinResult) * entry->capacity);
                    }
                    entry->results[entry->count].q_id = j;
                    entry->results[entry->count].c_id = c_id;
                    entry->count++;
                }
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
    for (size_t i = 0; i < total_levels; i++) if (all_L[i].k == k) { L = &all_L[i]; break; }
    if (!L) return 0;
    size_t low = 0, high = L->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = cmp_itemset(L->array[mid].items, items, k);
        if (cmp == 0) return L->array[mid].support;
        if (cmp < 0) low = mid + 1; else high = mid;
    }
    return 0;
}

static void gen_rules_hybrid(Apriori_Itemset l_k, Apriori_List *H_m, Apriori_List *all_L, size_t total_levels, double min_conf, size_t *rule_count) {
    size_t k = l_k.count, m = H_m->k;
    if (k > m + 1) {
        Apriori_List H_next;
        list_init(&H_next, m + 1);
        for (size_t i = 0; i < H_m->count; i++) {
            for (size_t j = i + 1; j < H_m->count; j++) {
                bool match = true;
                for (size_t x = 0; x < m - 1; x++) if (H_m->array[i].items[x] != H_m->array[j].items[x]) { match = false; break; }
                if (!match) break;
                uint32_t *c = (uint32_t*)malloc(sizeof(uint32_t) * (m + 1));
                memcpy(c, H_m->array[i].items, sizeof(uint32_t) * m);
                c[m] = H_m->array[j].items[m - 1];
                list_append(&H_next, c, m + 1, 0);
                free(c);
            }
        }
        for (size_t i = 0; i < H_next.count; ) {
            uint32_t *h = H_next.array[i].items;
            size_t ant_len = k - H_next.k;
            uint32_t *ant = (uint32_t*)malloc(sizeof(uint32_t) * ant_len);
            size_t ap = 0;
            for (size_t j = 0; j < k; j++) {
                bool in_h = false;
                for (size_t n = 0; n < H_next.k; n++) if (l_k.items[j] == h[n]) { in_h = true; break; }
                if (!in_h) ant[ap++] = l_k.items[j];
            }
            uint32_t ant_supp = get_support(all_L, total_levels, ant, ant_len);
            free(ant);
            if (ant_supp > 0 && (double)l_k.support / ant_supp >= min_conf) { (*rule_count)++; i++; }
            else {
                free(H_next.array[i].items);
                for (size_t j = i; j < H_next.count - 1; j++) H_next.array[j] = H_next.array[j+1];
                H_next.count--;
            }
        }
        if (H_next.count > 0) gen_rules_hybrid(l_k, &H_next, all_L, total_levels, min_conf, rule_count);
        list_free(&H_next);
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_APRIORI_HYBRID_Params *p = (DM_APRIORI_HYBRID_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    double min_conf = p ? p->min_confidence : 0.8;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[AprioriHybrid] Starting. Min Support: %u\n", min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);

    Apriori_List *all_L = (Apriori_List*)malloc(sizeof(Apriori_List) * 100);
    size_t levels = 0;

    // 1. Generate L1
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    list_init(&all_L[levels], 1);
    uint32_t *item_to_id = (uint32_t*)malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(item_to_id, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            item_to_id[i] = all_L[levels].count;
            uint32_t item = i;
            list_append(&all_L[levels], &item, 1, counts[i]);
        }
    }
    free(counts);
    printf("[AprioriHybrid] Found %zu 1-itemsets\n", all_L[levels].count);
    levels++;

    bool switch_to_tid = false;
    C_Prime_List C_prime = {0};

    // 2. Iterative Pass
    while (all_L[levels - 1].count > 0) {
        if (!switch_to_tid) {
            // Standard Apriori with Hash Tree
            Apriori_List C;
            apriori_gen_hybrid(&all_L[levels - 1], &C, NULL);
            if (C.count == 0) { list_free(&C); break; }
            HashTreeNode *root = create_leaf();
            for (size_t i = 0; i < C.count; i++) hash_tree_insert(root, &C.array[i], 0);
            for (size_t i = 0; i < ds->count; i++) if (data[i].count >= C.k) hash_tree_subset(root, data[i].items, data[i].count, 0, C.k);
            hash_tree_free(root);
            
            // Heuristic: Estimation for C'k size
            size_t total_support = 0;
            for (size_t i = 0; i < C.count; i++) total_support += C.array[i].support;
            size_t est_tid_size = total_support + ds->count;
            // If estimated size fits in memory (e.g., < 500MB), switch
            if (est_tid_size * sizeof(uint32_t) < 512 * 1024 * 1024) {
                printf("[AprioriHybrid] Switching to AprioriTid (est. size %zu words)\n", est_tid_size);
                switch_to_tid = true;
                // We need to build C_prime for the *current* L (which is L_k-1)
                C_prime.count = ds->count;
                C_prime.entries = (C_Prime_Entry*)malloc(sizeof(C_Prime_Entry) * ds->count);
                // We need item_to_id for the current level's items
                // Actually, the paper says we add IDs to C'_k during the scan of the pass we decide to switch.
                // But my current structure is simpler: build C'_k for the current L.
                // For level k, C'k contains IDs of candidates from Ck.
            }

            list_init(&all_L[levels], C.k);
            for (size_t i = 0; i < C.count; i++) if (C.array[i].support >= min_sup) list_append(&all_L[levels], C.array[i].items, C.k, C.array[i].support);
            
            if (switch_to_tid) {
                // To bootstrap Tid, we need C' for the NEW level
                // We scan once more to build C'k for the L[levels] we just found
                uint32_t *old_to_new_id = (uint32_t*)malloc(sizeof(uint32_t) * C.count);
                memset(old_to_new_id, 0xFF, sizeof(uint32_t) * C.count);
                // Wait, L[levels] items are copies of C items.
                // Let's just re-scan.
                for (size_t i = 0; i < ds->count; i++) {
                    C_prime.entries[i].tid = i;
                    C_prime.entries[i].count = 0;
                    C_prime.entries[i].capacity = 4;
                    C_prime.entries[i].ids = (uint32_t*)malloc(sizeof(uint32_t) * C_prime.entries[i].capacity);
                    for (size_t ci = 0; ci < C.count; ci++) {
                        if (C.array[ci].support >= min_sup && is_subset(C.array[ci].items, C.k, data[i].items, data[i].count)) {
                            if (C_prime.entries[i].count >= C_prime.entries[i].capacity) {
                                C_prime.entries[i].capacity *= 2;
                                C_prime.entries[i].ids = (uint32_t*)realloc(C_prime.entries[i].ids, sizeof(uint32_t) * C_prime.entries[i].capacity);
                            }
                            // Find the index in L[levels]
                            for (size_t li = 0; li < all_L[levels].count; li++) {
                                if (cmp_itemset(all_L[levels].array[li].items, C.array[ci].items, C.k) == 0) {
                                    C_prime.entries[i].ids[C_prime.entries[i].count++] = li;
                                    break;
                                }
                            }
                        }
                    }
                }
                free(old_to_new_id);
            }
            list_free(&C);
        } else {
            // AprioriTid logic
            Apriori_List C;
            JoinMapEntry *join_map = (JoinMapEntry*)calloc(all_L[levels - 1].count, sizeof(JoinMapEntry));
            apriori_gen_hybrid(&all_L[levels - 1], &C, join_map);
            if (C.count == 0) {
                for (size_t i = 0; i < all_L[levels-1].count; i++) if (join_map[i].results) free(join_map[i].results);
                free(join_map);
                list_free(&C);
                break;
            }
            C_Prime_List C_prime_next;
            C_prime_next.count = 0;
            C_prime_next.capacity = C_prime.count;
            C_prime_next.entries = (C_Prime_Entry*)malloc(sizeof(C_Prime_Entry) * C_prime_next.capacity);
            uint32_t *p_present = (uint32_t*)calloc(all_L[levels - 1].count, sizeof(uint32_t));
            uint32_t marker = 1;
            for (size_t i = 0; i < C_prime.count; i++) {
                C_Prime_Entry *t = &C_prime.entries[i];
                if (t->count < 2) continue;
                C_Prime_Entry t_next = {t->tid, NULL, 0, 4};
                t_next.ids = (uint32_t*)malloc(sizeof(uint32_t) * t_next.capacity);
                for (size_t j = 0; j < t->count; j++) p_present[t->ids[j]] = marker;
                for (size_t j = 0; j < t->count; j++) {
                    uint32_t p_id = t->ids[j];
                    JoinMapEntry *jm = &join_map[p_id];
                    for (size_t m = 0; m < jm->count; m++) {
                        if (p_present[jm->results[m].q_id] == marker) {
                            if (t_next.count >= t_next.capacity) {
                                t_next.capacity *= 2;
                                t_next.ids = (uint32_t*)realloc(t_next.ids, sizeof(uint32_t) * t_next.capacity);
                            }
                            t_next.ids[t_next.count++] = jm->results[m].c_id;
                            C.array[jm->results[m].c_id].support++;
                        }
                    }
                }
                marker++;
                if (t_next.count > 0) C_prime_next.entries[C_prime_next.count++] = t_next;
                else free(t_next.ids);
            }
            free(p_present);
            for (size_t i = 0; i < all_L[levels-1].count; i++) if (join_map[i].results) free(join_map[i].results);
            free(join_map);
            for (size_t i = 0; i < C_prime.count; i++) free(C_prime.entries[i].ids);
            free(C_prime.entries);
            C_prime = C_prime_next;
            list_init(&all_L[levels], C.k);
            uint32_t *old_to_new_id = (uint32_t*)malloc(sizeof(uint32_t) * C.count);
            memset(old_to_new_id, 0xFF, sizeof(uint32_t) * C.count);
            for (size_t i = 0; i < C.count; i++) {
                if (C.array[i].support >= min_sup) {
                    old_to_new_id[i] = all_L[levels].count;
                    list_append(&all_L[levels], C.array[i].items, C.k, C.array[i].support);
                }
            }
            list_free(&C);
            for (size_t i = 0; i < C_prime.count; i++) {
                size_t wr = 0;
                for (size_t j = 0; j < C_prime.entries[i].count; j++) {
                    uint32_t nid = old_to_new_id[C_prime.entries[i].ids[j]];
                    if (nid != 0xFFFFFFFF) C_prime.entries[i].ids[wr++] = nid;
                }
                C_prime.entries[i].count = wr;
            }
            free(old_to_new_id);
        }
        printf("[AprioriHybrid] Found %zu %zu-itemsets\n", all_L[levels].count, all_L[levels].k);
        if (all_L[levels].count == 0) break;
        levels++;
    }

    // Phase 3: Rule Gen
    size_t rule_count = 0;
    for (size_t i = 1; i < levels; i++) {
        for (size_t j = 0; j < all_L[i].count; j++) {
            Apriori_Itemset l_k = all_L[i].array[j];
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
                for (size_t m = 0; m < l_k.count; m++) if (l_k.items[m] != h1[0]) ant[ap++] = l_k.items[m];
                uint32_t ant_supp = get_support(all_L, levels, ant, ant_len);
                free(ant);
                if (ant_supp > 0 && (double)l_k.support / ant_supp >= min_conf) { rule_count++; h_idx++; }
                else {
                    free(H1.array[h_idx].items);
                    for (size_t m = h_idx; m < H1.count - 1; m++) H1.array[m] = H1.array[m+1];
                    H1.count--;
                }
            }
            if (H1.count > 0) gen_rules_hybrid(l_k, &H1, all_L, levels, min_conf, &rule_count);
            list_free(&H1);
        }
    }
    printf("[AprioriHybrid] Found %zu rules.\n", rule_count);

    size_t total_fi = 0, footprint = 0;
    for (size_t i = 0; i < levels; i++) {
        total_fi += all_L[i].count;
        footprint += all_L[i].count * all_L[i].k;
        list_free(&all_L[i]);
    }
    free(all_L);
    if (C_prime.entries) {
        for (size_t i = 0; i < C_prime.count; i++) free(C_prime.entries[i].ids);
        free(C_prime.entries);
    }
    free(item_to_id);

    dm_bench_record_results(total_fi, footprint);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "apriori_hybrid",
    .name = "AprioriHybrid Algorithm",
    .description = "Combines Apriori and AprioriTid as described in Agrawal & Srikant 1994.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
