#include "algorithms/uapriori.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

/**
 * U-Apriori Algorithm for Mining Frequent Itemsets from Uncertain Data.
 * Reference: C.-K. Chui, B. Kao, and E. Hung, "Mining Frequent Itemsets from Uncertain Data", PAKDD 2007.
 */

typedef struct {
    uint32_t *items;
    double support; // Expected Support
} UApriori_Itemset;

typedef struct {
    UApriori_Itemset *array;
    size_t count;
    size_t capacity;
    size_t k;
} UApriori_List;

static void list_init(UApriori_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (UApriori_Itemset *)malloc(sizeof(UApriori_Itemset) * list->capacity);
}

static void list_free(UApriori_List *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->array[i].items);
    }
    free(list->array);
}

static void list_append(UApriori_List *list, uint32_t *items, double support) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (UApriori_Itemset *)realloc(list->array, sizeof(UApriori_Itemset) * list->capacity);
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

static bool list_contains(UApriori_List *list, const uint32_t *items) {
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

static bool is_subset(const uint32_t *x, size_t x_len, size_t t_len, void *trans_items, DM_DatasetType type, double *prod_prob) {
    size_t i = 0, j = 0;
    *prod_prob = 1.0;
    
    if (type == DM_TYPE_TRANSACTIONAL) {
        uint32_t *items = (uint32_t *)trans_items;
        while (i < x_len && j < t_len) {
            if (x[i] == items[j]) { i++; j++; }
            else if (x[i] > items[j]) { j++; }
            else return false;
        }
    } else {
        DM_Item *items = (DM_Item *)trans_items;
        while (i < x_len && j < t_len) {
            if (x[i] == items[j].id) { 
                *prod_prob *= items[j].utility;
                i++; j++; 
            }
            else if (x[i] > items[j].id) { j++; }
            else return false;
        }
    }
    return i == x_len;
}

static void apriori_gen(UApriori_List *L_prev, UApriori_List *C_curr) {
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

            bool has_infrequent_subset = false;
            for (size_t drop_idx = 0; drop_idx < k; drop_idx++) {
                if (drop_idx == k - 1 || drop_idx == k - 2) continue;
                size_t s_idx = 0;
                for (size_t m = 0; m < k; m++) if (m != drop_idx) subset[s_idx++] = c_items[m];
                if (!list_contains(L_prev, subset)) {
                    has_infrequent_subset = true;
                    break;
                }
            }

            if (!has_infrequent_subset) list_append(C_curr, c_items, 0.0);
            else free(c_items);
        }
    }
    free(subset);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_UAPRIORI_Params *p = (DM_UAPRIORI_Params *)params;
    double min_sup_param = p ? p->min_support : 0.1;
    double expected_min_sup = (min_sup_param < 1.0) ? (min_sup_param * ds->count) : min_sup_param;

    printf("[U-Apriori] Starting on %zu transactions. Expected Min Support: %.2f\n", ds->count, expected_min_sup);

    // Prepare data (sorting is required for subset check)
    if (ds->type == DM_TYPE_TRANSACTIONAL) {
        DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
        for (size_t i = 0; i < ds->count; i++) qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);
    } else if (ds->type == DM_TYPE_UTILITY) {
        DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
        for (size_t i = 0; i < ds->count; i++) {
            for (size_t a = 0; a < data[i].count; a++) {
                for (size_t b = a + 1; b < data[i].count; b++) {
                    if (data[i].items[a].id > data[i].items[b].id) {
                        DM_Item tmp = data[i].items[a];
                        data[i].items[a] = data[i].items[b];
                        data[i].items[b] = tmp;
                    }
                }
            }
        }
    }

    // Step 1: L1
    double *counts = (double *)calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        if (ds->type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *tr = &((DM_Trans_Simple *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) counts[tr->items[j]] += 1.0;
        } else {
            DM_Trans_Utility *tr = &((DM_Trans_Utility *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) counts[tr->items[j].id] += tr->items[j].utility;
        }
    }

    UApriori_List L;
    list_init(&L, 1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= expected_min_sup - 1e-9) {
            uint32_t *item = (uint32_t *)malloc(sizeof(uint32_t));
            item[0] = i;
            list_append(&L, item, counts[i]);
        }
    }
    free(counts);

    size_t total_freq_itemsets = L.count;
    size_t total_items_footprint = L.count;
    printf("[U-Apriori] Found %zu 1-itemsets\n", L.count);

    size_t k = 2;
    while (L.count > 0) {
        UApriori_List C;
        apriori_gen(&L, &C);
        if (C.count == 0) { list_free(&C); break; }

        for (size_t i = 0; i < ds->count; i++) {
            void *tr_items;
            size_t tr_count;
            if (ds->type == DM_TYPE_TRANSACTIONAL) {
                DM_Trans_Simple *tr = &((DM_Trans_Simple *)ds->payload)[i];
                tr_items = tr->items; tr_count = tr->count;
            } else {
                DM_Trans_Utility *tr = &((DM_Trans_Utility *)ds->payload)[i];
                tr_items = tr->items; tr_count = tr->count;
            }
            if (tr_count < k) continue;

            for (size_t c = 0; c < C.count; c++) {
                double prob;
                if (is_subset(C.array[c].items, k, tr_count, tr_items, ds->type, &prob)) {
                    C.array[c].support += prob;
                }
            }
        }

        UApriori_List L_next;
        list_init(&L_next, k);
        for (size_t i = 0; i < C.count; i++) {
            if (C.array[i].support >= expected_min_sup - 1e-9) {
                uint32_t *freq_items = (uint32_t *)malloc(sizeof(uint32_t) * k);
                memcpy(freq_items, C.array[i].items, sizeof(uint32_t) * k);
                list_append(&L_next, freq_items, C.array[i].support);
            }
        }
        list_free(&C);

        if (L_next.count > 0) {
            printf("[U-Apriori] Found %zu %zu-itemsets\n", L_next.count, k);
            total_freq_itemsets += L_next.count;
            total_items_footprint += L_next.count * k;
        }
        list_free(&L);
        L = L_next;
        k++;
    }
    list_free(&L);

    printf("[U-Apriori] Complete. Total frequent itemsets found: %zu\n", total_freq_itemsets);
    dm_bench_record_results(total_freq_itemsets, total_items_footprint);
    return DM_SUCCESS;
}

static DM_Algorithm algo_uapriori = {
    .id = "uapriori",
    .name = "U-Apriori Algorithm",
    .description = "Frequent itemset mining from uncertain data using expected support.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL) | (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_uapriori)
