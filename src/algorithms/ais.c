#include "algorithms/ais.h"
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
} AIS_Itemset;

typedef struct {
    AIS_Itemset *array;
    size_t count;
    size_t capacity;
    size_t k;
} AIS_List;

static void list_init(AIS_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (AIS_Itemset *)malloc(sizeof(AIS_Itemset) * list->capacity);
}

static void list_free(AIS_List *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->array[i].items);
    }
    free(list->array);
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

static size_t list_find(AIS_List *list, const uint32_t *items, bool *found) {
    size_t low = 0, high = list->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = cmp_itemset(list->array[mid].items, items, list->k);
        if (cmp == 0) {
            *found = true;
            return mid;
        } else if (cmp < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    *found = false;
    return low;
}

static void list_add_or_inc(AIS_List *list, const uint32_t *items) {
    bool found;
    size_t pos = list_find(list, items, &found);
    if (found) {
        list->array[pos].support++;
    } else {
        if (list->count >= list->capacity) {
            list->capacity *= 2;
            list->array = (AIS_Itemset *)realloc(list->array, sizeof(AIS_Itemset) * list->capacity);
        }
        if (pos < list->count) {
            memmove(&list->array[pos + 1], &list->array[pos], sizeof(AIS_Itemset) * (list->count - pos));
        }
        list->array[pos].items = (uint32_t *)malloc(sizeof(uint32_t) * list->k);
        memcpy(list->array[pos].items, items, sizeof(uint32_t) * list->k);
        list->array[pos].support = 1;
        list->count++;
    }
}

static bool is_subset(const uint32_t *x, size_t x_len, const uint32_t *t, size_t t_len, size_t *start_ext_idx) {
    size_t i = 0, j = 0;
    size_t match_last_idx = 0;
    while (i < x_len && j < t_len) {
        if (x[i] == t[j]) {
            match_last_idx = j;
            i++; 
            j++; 
        }
        else if (x[i] > t[j]) { j++; }
        else { return false; }
    }
    if (i == x_len) {
        if (start_ext_idx) *start_ext_idx = match_last_idx + 1;
        return true;
    }
    return false;
}

/* --- ALGORITHM IMPLEMENTATION --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_AIS_Params *p = (DM_AIS_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01; // Default 1%
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[AIS] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    // 1. Sort items in every transaction to guarantee ascending order
    for (size_t i = 0; i < ds->count; i++) {
        qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);
    }

    // 2. Generate L1 (Frequent 1-itemsets)
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    AIS_List L;
    list_init(&L, 1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            if (L.count >= L.capacity) {
                L.capacity *= 2;
                L.array = (AIS_Itemset *)realloc(L.array, sizeof(AIS_Itemset) * L.capacity);
            }
            L.array[L.count].items = (uint32_t *)malloc(sizeof(uint32_t));
            L.array[L.count].items[0] = i;
            L.array[L.count].support = counts[i];
            L.count++;
        }
    }
    free(counts);

    size_t total_freq_itemsets = L.count;
    size_t total_item_count = L.count * 1; // k=1
    printf("[AIS] Found %zu 1-itemsets\n", L.count);

    // 3. Generate L_k for k >= 2
    size_t k = 2;
    uint32_t *cand_buf = (uint32_t *)malloc(sizeof(uint32_t) * (ds->max_id + 1));

    while (L.count > 0) {
        AIS_List C; // Candidates C_k
        list_init(&C, k);

        // Scan database on-the-fly
        for (size_t i = 0; i < ds->count; i++) {
            DM_Trans_Simple tr = data[i];
            if (tr.count < k) continue;

            for (size_t l_idx = 0; l_idx < L.count; l_idx++) {
                size_t start_ext_idx;
                if (is_subset(L.array[l_idx].items, k - 1, tr.items, tr.count, &start_ext_idx)) {
                    // Extend subset with the remaining items in transaction
                    for (size_t j = start_ext_idx; j < tr.count; j++) {
                        memcpy(cand_buf, L.array[l_idx].items, sizeof(uint32_t) * (k - 1));
                        cand_buf[k - 1] = tr.items[j];
                        list_add_or_inc(&C, cand_buf);
                    }
                }
            }
        }

        // Filter C_k to form L_k
        AIS_List L_next;
        list_init(&L_next, k);
        for (size_t i = 0; i < C.count; i++) {
            if (C.array[i].support >= min_sup) {
                if (L_next.count >= L_next.capacity) {
                    L_next.capacity *= 2;
                    L_next.array = (AIS_Itemset *)realloc(L_next.array, sizeof(AIS_Itemset) * L_next.capacity);
                }
                L_next.array[L_next.count].items = (uint32_t *)malloc(sizeof(uint32_t) * k);
                memcpy(L_next.array[L_next.count].items, C.array[i].items, sizeof(uint32_t) * k);
                L_next.array[L_next.count].support = C.array[i].support;
                L_next.count++;
            }
        }

        list_free(&C);
        
        if (L_next.count > 0) {
            printf("[AIS] Found %zu %zu-itemsets\n", L_next.count, k);
            total_freq_itemsets += L_next.count;
            total_item_count += L_next.count * k;
        }

        list_free(&L);
        L = L_next;
        k++;
    }

    list_free(&L);
    free(cand_buf);
    
    printf("[AIS] Complete. Total frequent itemsets found: %zu\n", total_freq_itemsets);

    // Record footprint metrics
    dm_bench_record_results(total_freq_itemsets, total_item_count);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "ais",
    .name = "AIS Algorithm",
    .description = "On-the-fly candidate generation algorithm for association rules.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
