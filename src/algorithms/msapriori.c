#include "algorithms/msapriori.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

/**
 * MSApriori (Multiple Minimum Support Apriori) Algorithm.
 * Reference: Bing Liu, Wynne Hsu and Yiming Ma, "Mining Association Rules with Multiple Minimum Supports", KDD 1999.
 */

typedef struct {
    uint32_t *items;
    uint32_t count;
} MSApriori_Itemset;

typedef struct {
    MSApriori_Itemset *array;
    size_t count;
    size_t capacity;
    size_t k;
} MSApriori_List;

static void list_init(MSApriori_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (MSApriori_Itemset *)malloc(sizeof(MSApriori_Itemset) * list->capacity);
}

static void list_free(MSApriori_List *list) {
    for (size_t i = 0; i < list->count; i++) free(list->array[i].items);
    free(list->array);
}

static void list_append(MSApriori_List *list, uint32_t *items, uint32_t count) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (MSApriori_Itemset *)realloc(list->array, sizeof(MSApriori_Itemset) * list->capacity);
    }
    list->array[list->count].items = items;
    list->array[list->count].count = count;
    list->count++;
}

static int cmp_itemset(const uint32_t *a, const uint32_t *b, size_t k) {
    for (size_t i = 0; i < k; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static bool list_contains(MSApriori_List *list, const uint32_t *items) {
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
        else if (x[i] > t[j]) j++;
        else return false;
    }
    return i == x_len;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

// Custom sort for items based on MIS values
typedef struct {
    uint32_t id;
    double mis;
} ItemMIS;

static int cmp_item_mis(const void *a, const void *b) {
    const ItemMIS *x = (const ItemMIS *)a;
    const ItemMIS *y = (const ItemMIS *)b;
    if (x->mis < y->mis) return -1;
    if (x->mis > y->mis) return 1;
    return (x->id < y->id) ? -1 : (x->id > y->id);
}

/* --- MSApriori Functions --- */

static void level2_candidate_gen(uint32_t *F, uint32_t F_count, uint32_t *item_counts, double *MIS, MSApriori_List *C2) {
    list_init(C2, 2);
    for (uint32_t i = 0; i < F_count; i++) {
        uint32_t f = F[i];
        if (item_counts[f] >= (uint32_t)ceil(MIS[f])) {
            for (uint32_t j = i + 1; j < F_count; j++) {
                uint32_t h = F[j];
                if (item_counts[h] >= (uint32_t)ceil(MIS[f])) {
                    uint32_t *c = malloc(sizeof(uint32_t) * 2);
                    c[0] = f; c[1] = h;
                    list_append(C2, c, 0);
                }
            }
        }
    }
}

static void candidate_gen(MSApriori_List *L_prev, double *MIS, MSApriori_List *C_curr) {
    size_t k = L_prev->k + 1;
    list_init(C_curr, k);
    uint32_t *subset = malloc(sizeof(uint32_t) * (k - 1));

    for (size_t i = 0; i < L_prev->count; i++) {
        for (size_t j = i + 1; j < L_prev->count; j++) {
            // Join condition: first k-2 items must be identical
            bool match = true;
            for (size_t m = 0; m < k - 2; m++) {
                if (L_prev->array[i].items[m] != L_prev->array[j].items[m]) {
                    match = false; break;
                }
            }
            if (!match) break;

            // p[k-2] < q[k-2] (last items)
            if (L_prev->array[i].items[k - 2] >= L_prev->array[j].items[k - 2]) continue;

            // Generate candidate c
            uint32_t *c = malloc(sizeof(uint32_t) * k);
            memcpy(c, L_prev->array[i].items, sizeof(uint32_t) * (k - 1));
            c[k - 1] = L_prev->array[j].items[k - 2];

            // Prune Step
            bool pruned = false;
            for (size_t drop_idx = 0; drop_idx < k; drop_idx++) {
                // Generate (k-1)-subset
                size_t s_idx = 0;
                for (size_t m = 0; m < k; m++) if (m != drop_idx) subset[s_idx++] = c[m];

                // Rule: If subset contains c[0] OR MIS(c[1]) == MIS(c[0])
                // Then subset must be in L_prev
                if (drop_idx != 0 || MIS[c[1]] == MIS[c[0]]) {
                    if (!list_contains(L_prev, subset)) {
                        pruned = true; break;
                    }
                }
            }

            if (!pruned) list_append(C_curr, c, 0);
            else free(c);
        }
    }
    free(subset);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_MSAPRIORI_Params *p = (DM_MSAPRIORI_Params *)params;
    double beta = p ? p->beta : 1.0;
    double LS = p ? p->LS : 0.01;
    double abs_LS = LS * ds->count;

    printf("[MSApriori] Starting on %zu transactions. LS: %.4f, Beta: %.2f\n", ds->count, LS, beta);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);

    // 1. Pass 1: Count item supports
    uint32_t *item_counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) item_counts[data[i].items[j]]++;
    }

    // 2. Assign MIS values if not provided
    double *MIS = malloc(sizeof(double) * (ds->max_id + 1));
    if (p && p->mis_values) {
        memcpy(MIS, p->mis_values, sizeof(double) * (ds->max_id + 1));
    } else {
        for (uint32_t i = 0; i <= ds->max_id; i++) {
            double sup = (double)item_counts[i] / ds->count;
            MIS[i] = (beta * sup < LS) ? LS : (beta * sup);
            MIS[i] *= ds->count; // Work with absolute counts for convenience
        }
    }

    // 3. Sort items according to MIS values (M)
    ItemMIS *M_temp = malloc(sizeof(ItemMIS) * (ds->max_id + 1));
    uint32_t M_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_counts[i] > 0) {
            M_temp[M_count].id = i;
            M_temp[M_count].mis = MIS[i];
            M_count++;
        }
    }
    qsort(M_temp, M_count, sizeof(ItemMIS), cmp_item_mis);

    // 4. Init-pass: Generate F
    uint32_t *F = malloc(sizeof(uint32_t) * M_count);
    uint32_t F_count = 0;
    int first_idx = -1;
    for (uint32_t i = 0; i < M_count; i++) {
        if (item_counts[M_temp[i].id] >= MIS[M_temp[i].id]) {
            first_idx = i;
            F[F_count++] = M_temp[i].id;
            break;
        }
    }
    if (first_idx != -1) {
        double mis_first = MIS[M_temp[first_idx].id];
        for (uint32_t i = first_idx + 1; i < M_count; i++) {
            if (item_counts[M_temp[i].id] >= mis_first) {
                F[F_count++] = M_temp[i].id;
            }
        }
    }

    // 5. Generate L1
    MSApriori_List L;
    list_init(&L, 1);
    for (uint32_t i = 0; i < F_count; i++) {
        uint32_t id = F[i];
        if (item_counts[id] >= MIS[id]) {
            uint32_t *item = malloc(sizeof(uint32_t));
            item[0] = id;
            list_append(&L, item, item_counts[id]);
        }
    }

    size_t total_freq = L.count;
    size_t total_footprint = L.count;
    printf("[MSApriori] Found %zu 1-itemsets\n", L.count);

    // 6. Generate L_k (k >= 2)
    size_t k = 2;
    while (L.count > 0) {
        MSApriori_List C;
        if (k == 2) level2_candidate_gen(F, F_count, item_counts, MIS, &C);
        else candidate_gen(&L, MIS, &C);

        if (C.count == 0) { list_free(&C); break; }

        // Support counting
        for (size_t i = 0; i < ds->count; i++) {
            if (data[i].count < k) continue;
            for (size_t c = 0; c < C.count; c++) {
                if (is_subset(C.array[c].items, k, data[i].items, data[i].count)) {
                    C.array[c].count++;
                }
            }
        }

        // Filter C_k to L_k
        MSApriori_List L_next;
        list_init(&L_next, k);
        for (size_t i = 0; i < C.count; i++) {
            if (C.array[i].count >= (uint32_t)ceil(MIS[C.array[i].items[0]])) {
                uint32_t *items = malloc(sizeof(uint32_t) * k);
                memcpy(items, C.array[i].items, sizeof(uint32_t) * k);
                list_append(&L_next, items, C.array[i].count);
            }
        }
        list_free(&C);

        if (L_next.count > 0) {
            printf("[MSApriori] Found %zu %zu-itemsets\n", L_next.count, k);
            total_freq += L_next.count;
            total_footprint += L_next.count * k;
        }
        list_free(&L);
        L = L_next;
        k++;
    }

    list_free(&L);
    free(F); free(M_temp); free(MIS); free(item_counts);

    printf("[MSApriori] Complete. Total found: %zu\n", total_freq);
    dm_bench_record_results(total_freq, total_footprint);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "msapriori",
    .name = "MSApriori Algorithm",
    .description = "Mining with Multiple Minimum Supports (Liu et al. 1999).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
