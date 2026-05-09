#include "algorithms/apriori.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/* --- INTERNAL DATA STRUCTURES --- */

typedef struct {
    uint32_t *items;
    uint32_t support;
} Apriori_Itemset;

typedef struct {
    Apriori_Itemset *array;
    size_t count;
    size_t capacity;
    size_t k;
} Apriori_List;

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

static void list_append(Apriori_List *list, uint32_t *items, uint32_t support) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (Apriori_Itemset *)realloc(list->array, sizeof(Apriori_Itemset) * list->capacity);
    }
    list->array[list->count].items = items;
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

static bool is_subset(const uint32_t *x, size_t x_len, const uint32_t *t, size_t t_len) {
    size_t i = 0, j = 0;
    while (i < x_len && j < t_len) {
        if (x[i] == t[j]) { i++; j++; }
        else if (x[i] > t[j]) { j++; }
        else { return false; }
    }
    return i == x_len;
}

/* --- APRIORI-GEN (Join & Prune) --- */
static void apriori_gen(Apriori_List *L_prev, Apriori_List *C_curr) {
    size_t k = L_prev->k + 1;
    list_init(C_curr, k);
    
    uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (k - 1));

    for (size_t i = 0; i < L_prev->count; i++) {
        for (size_t j = i + 1; j < L_prev->count; j++) {
            // Join condition: items 0 to k-3 must be identical
            bool match = true;
            for (size_t m = 0; m < k - 2; m++) {
                if (L_prev->array[i].items[m] != L_prev->array[j].items[m]) {
                    match = false;
                    break;
                }
            }
            if (!match) break; // Due to lexicographical sorting, no further j will match this prefix

            // Generate candidate c = L_prev[i] U {L_prev[j][last]}
            uint32_t *c_items = (uint32_t *)malloc(sizeof(uint32_t) * k);
            memcpy(c_items, L_prev->array[i].items, sizeof(uint32_t) * (k - 1));
            c_items[k - 1] = L_prev->array[j].items[k - 2];

            // Prune Step: Ensure all (k-1)-subsets are in L_prev
            bool has_infrequent_subset = false;
            for (size_t drop_idx = 0; drop_idx < k; drop_idx++) {
                if (drop_idx == k - 1 || drop_idx == k - 2) continue; // Optimization: these subsets are exactly L_prev[i] and L_prev[j]
                
                size_t s_idx = 0;
                for (size_t m = 0; m < k; m++) {
                    if (m != drop_idx) subset[s_idx++] = c_items[m];
                }
                
                if (!list_contains(L_prev, subset)) {
                    has_infrequent_subset = true;
                    break;
                }
            }

            if (!has_infrequent_subset) {
                list_append(C_curr, c_items, 0); // Resulting list is naturally sorted!
            } else {
                free(c_items);
            }
        }
    }
    free(subset);
}

/* --- MAIN ALGORITHM --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_APRIORI_Params *p = (DM_APRIORI_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[Apriori] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    // 1. Sort items in every transaction
    for (size_t i = 0; i < ds->count; i++) {
        qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);
    }

    // 2. Generate L1
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    Apriori_List L;
    list_init(&L, 1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            uint32_t *item = (uint32_t *)malloc(sizeof(uint32_t));
            item[0] = i;
            list_append(&L, item, counts[i]);
        }
    }
    free(counts);

    size_t total_freq_itemsets = L.count;
    size_t total_items_footprint = L.count * 1;
    printf("[Apriori] Found %zu 1-itemsets\n", L.count);

    // 3. Generate L_k (k >= 2)
    size_t k = 2;
    while (L.count > 0) {
        Apriori_List C;
        apriori_gen(&L, &C);

        if (C.count == 0) {
            list_free(&C);
            break;
        }

        // Count supports by scanning database
        for (size_t i = 0; i < ds->count; i++) {
            DM_Trans_Simple tr = data[i];
            if (tr.count < k) continue;
            
            for (size_t c = 0; c < C.count; c++) {
                if (is_subset(C.array[c].items, k, tr.items, tr.count)) {
                    C.array[c].support++;
                }
            }
        }

        // Filter C_k to L_k
        Apriori_List L_next;
        list_init(&L_next, k);
        for (size_t i = 0; i < C.count; i++) {
            if (C.array[i].support >= min_sup) {
                uint32_t *freq_items = (uint32_t *)malloc(sizeof(uint32_t) * k);
                memcpy(freq_items, C.array[i].items, sizeof(uint32_t) * k);
                list_append(&L_next, freq_items, C.array[i].support);
            }
        }

        list_free(&C);
        
        if (L_next.count > 0) {
            printf("[Apriori] Found %zu %zu-itemsets\n", L_next.count, k);
            total_freq_itemsets += L_next.count;
            total_items_footprint += L_next.count * k;
        }

        list_free(&L);
        L = L_next;
        k++;
    }

    list_free(&L);
    
    printf("[Apriori] Complete. Total frequent itemsets found: %zu\n", total_freq_itemsets);
    dm_bench_record_results(total_freq_itemsets, total_items_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "apriori",
    .name = "Apriori Algorithm",
    .description = "Exact implementation from Agrawal & Srikant 1994.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
