#include "algorithms/aclose.h"
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
    uint32_t *closure;
    size_t closure_count;
} AClose_Generator;

typedef struct {
    AClose_Generator *array;
    size_t count;
    size_t capacity;
    size_t k;
} AClose_List;

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t support;
} ClosedItemset;

static void list_init(AClose_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (AClose_Generator *)malloc(sizeof(AClose_Generator) * list->capacity);
}

static void list_free(AClose_List *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->array[i].items);
        if (list->array[i].closure) {
            free(list->array[i].closure);
        }
    }
    free(list->array);
}

static void list_append(AClose_List *list, uint32_t *items, uint32_t support) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (AClose_Generator *)realloc(list->array, sizeof(AClose_Generator) * list->capacity);
    }
    list->array[list->count].items = items;
    list->array[list->count].count = list->k;
    list->array[list->count].support = support;
    list->array[list->count].closure = NULL;
    list->array[list->count].closure_count = 0;
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

static long list_index_of(AClose_List *list, const uint32_t *items) {
    size_t low = 0, high = list->count;
    size_t k = list->k;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = cmp_itemset(list->array[mid].items, items, k);
        if (cmp == 0) return (long)mid;
        if (cmp < 0) low = mid + 1;
        else high = mid;
    }
    return -1;
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

static void aclose_gen(AClose_List *L_prev, AClose_List *C_curr) {
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
                for (size_t m = 0; m < k; m++) {
                    if (m != drop_idx) subset[s_idx++] = c_items[m];
                }
                
                if (list_index_of(L_prev, subset) < 0) {
                    has_infrequent_subset = true;
                    break;
                }
            }

            if (!has_infrequent_subset) {
                list_append(C_curr, c_items, 0); 
            } else {
                free(c_items);
            }
        }
    }
    free(subset);
}

int cmp_closed_itemset(const void *a, const void *b) {
    const ClosedItemset *ca = (const ClosedItemset *)a;
    const ClosedItemset *cb = (const ClosedItemset *)b;
    if (ca->count < cb->count) return -1;
    if (ca->count > cb->count) return 1;
    for (size_t i = 0; i < ca->count; i++) {
        if (ca->items[i] < cb->items[i]) return -1;
        if (ca->items[i] > cb->items[i]) return 1;
    }
    return 0;
}

/* --- MAIN ALGORITHM --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_ACLOSE_Params *p = (DM_ACLOSE_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[A-Close] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    for (size_t i = 0; i < ds->count; i++) {
        qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);
    }

    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    AClose_List *all_lists = NULL;
    size_t all_lists_count = 0;
    size_t all_lists_cap = 16;
    all_lists = (AClose_List *)malloc(sizeof(AClose_List) * all_lists_cap);
    
    AClose_List L1;
    list_init(&L1, 1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            uint32_t *item = (uint32_t *)malloc(sizeof(uint32_t));
            item[0] = i;
            list_append(&L1, item, counts[i]);
        }
    }
    free(counts);

    all_lists[0] = L1;
    all_lists_count = 1;

    printf("[A-Close] Found %zu 1-generators\n", L1.count);

    size_t k = 2;
    size_t level = 0;
    
    while (true) {
        AClose_List *L_prev = &all_lists[all_lists_count - 1];
        if (L_prev->count == 0) break;
        
        AClose_List C;
        aclose_gen(L_prev, &C);

        if (C.count == 0) {
            list_free(&C);
            break;
        }

        for (size_t i = 0; i < ds->count; i++) {
            DM_Trans_Simple tr = data[i];
            if (tr.count < k) continue;
            
            for (size_t c = 0; c < C.count; c++) {
                if (is_subset(C.array[c].items, k, tr.items, tr.count)) {
                    C.array[c].support++;
                }
            }
        }

        AClose_List L_next;
        list_init(&L_next, k);
        
        uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (k - 1));
        
        for (size_t i = 0; i < C.count; i++) {
            if (C.array[i].support >= min_sup) {
                bool same_support = false;
                for (size_t drop_idx = 0; drop_idx < k; drop_idx++) {
                    size_t s_idx = 0;
                    for (size_t m = 0; m < k; m++) {
                        if (m != drop_idx) subset[s_idx++] = C.array[i].items[m];
                    }
                    long sub_idx = list_index_of(L_prev, subset);
                    if (sub_idx >= 0 && L_prev->array[sub_idx].support == C.array[i].support) {
                        same_support = true;
                        break;
                    }
                }
                
                if (!same_support) {
                    uint32_t *freq_items = (uint32_t *)malloc(sizeof(uint32_t) * k);
                    memcpy(freq_items, C.array[i].items, sizeof(uint32_t) * k);
                    list_append(&L_next, freq_items, C.array[i].support);
                } else {
                    if (level == 0) level = k - 1; 
                }
            }
        }
        free(subset);
        list_free(&C);
        
        if (all_lists_count == all_lists_cap) {
            all_lists_cap *= 2;
            all_lists = (AClose_List *)realloc(all_lists, sizeof(AClose_List) * all_lists_cap);
        }
        all_lists[all_lists_count++] = L_next;
        
        if (L_next.count > 0) {
            printf("[A-Close] Found %zu %zu-generators\n", L_next.count, k);
        }
        k++;
    }

    if (level != 0) {
        for (size_t t = 0; t < ds->count; t++) {
            DM_Trans_Simple tr = data[t];
            for (size_t k_idx = 0; k_idx < all_lists_count; k_idx++) {
                AClose_List *L_curr = &all_lists[k_idx];
                size_t curr_k = L_curr->k;
                if (level > 2 && curr_k < level - 1) continue; 

                for (size_t i = 0; i < L_curr->count; i++) {
                    if (is_subset(L_curr->array[i].items, curr_k, tr.items, tr.count)) {
                        if (L_curr->array[i].closure == NULL) {
                            L_curr->array[i].closure = (uint32_t *)malloc(sizeof(uint32_t) * tr.count);
                            memcpy(L_curr->array[i].closure, tr.items, sizeof(uint32_t) * tr.count);
                            L_curr->array[i].closure_count = tr.count;
                        } else {
                            size_t c_idx = 0, t_idx = 0, new_count = 0;
                            while (c_idx < L_curr->array[i].closure_count && t_idx < tr.count) {
                                if (L_curr->array[i].closure[c_idx] == tr.items[t_idx]) {
                                    L_curr->array[i].closure[new_count++] = tr.items[t_idx];
                                    c_idx++; t_idx++;
                                } else if (L_curr->array[i].closure[c_idx] < tr.items[t_idx]) {
                                    c_idx++;
                                } else {
                                    t_idx++;
                                }
                            }
                            L_curr->array[i].closure_count = new_count;
                        }
                    }
                }
            }
        }
        if (level > 2) {
            for (size_t k_idx = 0; k_idx < all_lists_count; k_idx++) {
                AClose_List *L_curr = &all_lists[k_idx];
                size_t curr_k = L_curr->k;
                if (curr_k < level - 1) {
                    for (size_t i = 0; i < L_curr->count; i++) {
                        L_curr->array[i].closure = (uint32_t *)malloc(sizeof(uint32_t) * curr_k);
                        memcpy(L_curr->array[i].closure, L_curr->array[i].items, sizeof(uint32_t) * curr_k);
                        L_curr->array[i].closure_count = curr_k;
                    }
                }
            }
        }
    } else {
        for (size_t k_idx = 0; k_idx < all_lists_count; k_idx++) {
            AClose_List *L_curr = &all_lists[k_idx];
            size_t curr_k = L_curr->k;
            for (size_t i = 0; i < L_curr->count; i++) {
                L_curr->array[i].closure = (uint32_t *)malloc(sizeof(uint32_t) * curr_k);
                memcpy(L_curr->array[i].closure, L_curr->array[i].items, sizeof(uint32_t) * curr_k);
                L_curr->array[i].closure_count = curr_k;
            }
        }
    }

    size_t total_closures = 0;
    for (size_t k_idx = 0; k_idx < all_lists_count; k_idx++) {
        total_closures += all_lists[k_idx].count;
    }

    ClosedItemset *fc = (ClosedItemset *)malloc(sizeof(ClosedItemset) * total_closures);
    size_t fc_count = 0;

    for (size_t k_idx = 0; k_idx < all_lists_count; k_idx++) {
        AClose_List *L_curr = &all_lists[k_idx];
        for (size_t i = 0; i < L_curr->count; i++) {
            if (L_curr->array[i].closure != NULL) {
                fc[fc_count].items = L_curr->array[i].closure;
                fc[fc_count].count = L_curr->array[i].closure_count;
                fc[fc_count].support = L_curr->array[i].support;
                fc_count++;
            }
        }
    }

    qsort(fc, fc_count, sizeof(ClosedItemset), cmp_closed_itemset);

    size_t unique_fc_count = 0;
    size_t footprint = 0;
    if (fc_count > 0) {
        unique_fc_count = 1;
        footprint += fc[0].count;
        for (size_t i = 1; i < fc_count; i++) {
            if (cmp_closed_itemset(&fc[i - 1], &fc[i]) != 0) {
                unique_fc_count++;
                footprint += fc[i].count;
            }
        }
    }

    printf("[A-Close] Complete. Total frequent closed itemsets found: %zu\n", unique_fc_count);
    dm_bench_record_results(unique_fc_count, footprint);

    free(fc);
    for (size_t k_idx = 0; k_idx < all_lists_count; k_idx++) {
        list_free(&all_lists[k_idx]);
    }
    free(all_lists);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "aclose",
    .name = "A-Close Algorithm",
    .description = "Discovering frequent closed itemsets (Pasquier et al. 1999).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
