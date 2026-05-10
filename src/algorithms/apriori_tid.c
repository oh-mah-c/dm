#include "algorithms/apriori_tid.h"
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

/* --- UTILS --- */

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

/* --- APRIORI-GEN FOR TID --- */
static void apriori_gen_tid(Apriori_List *L_prev, Apriori_List *C_curr, JoinMapEntry *join_map) {
    size_t k = L_prev->k + 1;
    list_init(C_curr, k);
    
    uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (k - 1));

    for (size_t i = 0; i < L_prev->count; i++) {
        for (size_t j = i + 1; j < L_prev->count; j++) {
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
                uint32_t c_id = C_curr->count;
                list_append(C_curr, c_items, k, 0);
                
                // Add to join map: (p_id, q_id) -> c_id
                JoinMapEntry *entry = &join_map[i];
                if (entry->count >= entry->capacity) {
                    entry->capacity = entry->capacity == 0 ? 4 : entry->capacity * 2;
                    entry->results = (JoinResult*)realloc(entry->results, sizeof(JoinResult) * entry->capacity);
                }
                entry->results[entry->count].q_id = j;
                entry->results[entry->count].c_id = c_id;
                entry->count++;
            }
            free(c_items);
        }
    }
    free(subset);
}

/* --- RULE GENERATION (SAME AS APRIORI) --- */

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
        // We need a simple join for H_next
        list_init(&H_next, m + 1);
        uint32_t *subset = (uint32_t*)malloc(sizeof(uint32_t) * m);
        for (size_t i = 0; i < H_m->count; i++) {
            for (size_t j = i + 1; j < H_m->count; j++) {
                bool match = true;
                for (size_t x = 0; x < m - 1; x++) {
                    if (H_m->array[i].items[x] != H_m->array[j].items[x]) { match = false; break; }
                }
                if (!match) break;
                uint32_t *c = (uint32_t*)malloc(sizeof(uint32_t) * (m + 1));
                memcpy(c, H_m->array[i].items, sizeof(uint32_t) * m);
                c[m] = H_m->array[j].items[m - 1];
                list_append(&H_next, c, m + 1, 0);
                free(c);
            }
        }
        free(subset);

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
                free(H_next.array[i].items);
                for (size_t j = i; j < H_next.count - 1; j++) H_next.array[j] = H_next.array[j+1];
                H_next.count--;
            }
        }
        if (H_next.count > 0) gen_rules(l_k, &H_next, all_L, total_levels, min_conf, rule_count);
        list_free(&H_next);
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_APRIORI_TID_Params *p = (DM_APRIORI_TID_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    double min_conf = p ? p->min_confidence : 0.8;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[AprioriTid] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);

    Apriori_List *all_L = (Apriori_List*)malloc(sizeof(Apriori_List) * 100);
    size_t levels = 0;

    // 1. Generate L1 and C'1
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

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
    printf("[AprioriTid] Found %zu 1-itemsets\n", all_L[levels].count);

    C_Prime_List C_prime;
    C_prime.count = ds->count;
    C_prime.entries = (C_Prime_Entry*)malloc(sizeof(C_Prime_Entry) * ds->count);
    for (size_t i = 0; i < ds->count; i++) {
        C_prime.entries[i].tid = i;
        C_prime.entries[i].capacity = data[i].count;
        C_prime.entries[i].ids = (uint32_t*)malloc(sizeof(uint32_t) * C_prime.entries[i].capacity);
        C_prime.entries[i].count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = item_to_id[data[i].items[j]];
            if (id != 0xFFFFFFFF) C_prime.entries[i].ids[C_prime.entries[i].count++] = id;
        }
    }
    free(item_to_id);
    levels++;

    // 2. Pass k > 1
    while (all_L[levels - 1].count > 0) {
        Apriori_List C;
        JoinMapEntry *join_map = (JoinMapEntry*)calloc(all_L[levels - 1].count, sizeof(JoinMapEntry));
        apriori_gen_tid(&all_L[levels - 1], &C, join_map);
        if (C.count == 0) {
            for (size_t i = 0; i < all_L[levels-1].count; i++) if (join_map[i].results) free(join_map[i].results);
            free(join_map);
            list_free(&C);
            break;
        }

        // Generate C'k from C'k-1
        C_Prime_List C_prime_next;
        C_prime_next.count = 0;
        C_prime_next.capacity = C_prime.count;
        C_prime_next.entries = (C_Prime_Entry*)malloc(sizeof(C_Prime_Entry) * C_prime_next.capacity);

        uint32_t *p_present = (uint32_t*)malloc(sizeof(uint32_t) * all_L[levels - 1].count);
        memset(p_present, 0, sizeof(uint32_t) * all_L[levels - 1].count);
        uint32_t marker = 1;

        for (size_t i = 0; i < C_prime.count; i++) {
            C_Prime_Entry *t = &C_prime.entries[i];
            if (t->count < 2) continue;

            C_Prime_Entry t_next;
            t_next.tid = t->tid;
            t_next.count = 0;
            t_next.capacity = 4;
            t_next.ids = (uint32_t*)malloc(sizeof(uint32_t) * t_next.capacity);

            // Mark which p's are in this transaction
            for (size_t j = 0; j < t->count; j++) p_present[t->ids[j]] = marker;

            for (size_t j = 0; j < t->count; j++) {
                uint32_t p_id = t->ids[j];
                JoinMapEntry *jm = &join_map[p_id];
                for (size_t m = 0; m < jm->count; m++) {
                    if (p_present[jm->results[m].q_id] == marker) {
                        // Candidate c is in transaction t!
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

            if (t_next.count > 0) {
                C_prime_next.entries[C_prime_next.count++] = t_next;
            } else {
                free(t_next.ids);
            }
        }
        free(p_present);

        // Cleanup join map and old C_prime
        for (size_t i = 0; i < all_L[levels-1].count; i++) if (join_map[i].results) free(join_map[i].results);
        free(join_map);
        for (size_t i = 0; i < C_prime.count; i++) free(C_prime.entries[i].ids);
        free(C_prime.entries);
        C_prime = C_prime_next;

        // Filter Lk
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

        // Update IDs in C'k to match Lk indices
        for (size_t i = 0; i < C_prime.count; i++) {
            size_t write_idx = 0;
            for (size_t j = 0; j < C_prime.entries[i].count; j++) {
                uint32_t new_id = old_to_new_id[C_prime.entries[i].ids[j]];
                if (new_id != 0xFFFFFFFF) C_prime.entries[i].ids[write_idx++] = new_id;
            }
            C_prime.entries[i].count = write_idx;
        }
        free(old_to_new_id);

        printf("[AprioriTid] Found %zu %zu-itemsets\n", all_L[levels].count, all_L[levels].k);
        if (all_L[levels].count == 0) break;
        levels++;
    }

    // Phase 2: Rule Gen (same logic)
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
            if (H1.count > 0) gen_rules(l_k, &H1, all_L, levels, min_conf, &rule_count);
            list_free(&H1);
        }
    }
    printf("[AprioriTid] Found %zu rules.\n", rule_count);

    size_t total_fi = 0, footprint = 0;
    for (size_t i = 0; i < levels; i++) {
        total_fi += all_L[i].count;
        footprint += all_L[i].count * all_L[i].k;
        list_free(&all_L[i]);
    }
    free(all_L);
    for (size_t i = 0; i < C_prime.count; i++) free(C_prime.entries[i].ids);
    free(C_prime.entries);

    dm_bench_record_results(total_fi, footprint);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "apriori_tid",
    .name = "AprioriTid Algorithm",
    .description = "Efficient implementation from Agrawal & Srikant 1994 using candidate ID sets.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
