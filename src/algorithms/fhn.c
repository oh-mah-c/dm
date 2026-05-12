#include "algorithms/fhn.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iputil; // Utility of positive items in X
    double inutil; // Utility of negative items in X
    double rutil;  // Sum of utilities of positive items after X
} UtilityTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    UtilityTuple *tuples;
    size_t tuple_count;
    double sum_iputil;
    double sum_inutil;
    double sum_rutil;
} UtilityList;

typedef struct {
    uint32_t u, v;
    double twu;
} EUCS_Entry;

typedef struct {
    EUCS_Entry *entries;
    size_t count;
    size_t capacity;
} EUCS;

/* --- UTILS --- */

typedef struct {
    uint32_t id;
    double twu;
    bool is_positive;
} ItemInfo;

static int cmp_item_info(const void *a, const void *b) {
    ItemInfo *ia = (ItemInfo *)a;
    ItemInfo *ib = (ItemInfo *)b;
    
    // Positive items before Negative items
    if (ia->is_positive && !ib->is_positive) return -1;
    if (!ia->is_positive && ib->is_positive) return 1;
    
    // Among same type: ascending TWU
    if (ia->twu < ib->twu) return -1;
    if (ia->twu > ib->twu) return 1;
    
    return (int)(ia->id - ib->id);
}

/* --- EUCS --- */

static void eucs_add(EUCS *eucs, uint32_t u, uint32_t v, double twu) {
    if (u > v) { uint32_t t = u; u = v; v = t; }
    for (size_t i = 0; i < eucs->count; i++) {
        if (eucs->entries[i].u == u && eucs->entries[i].v == v) {
            eucs->entries[i].twu += twu;
            return;
        }
    }
    if (eucs->count >= eucs->capacity) {
        eucs->capacity *= 2;
        eucs->entries = realloc(eucs->entries, sizeof(EUCS_Entry) * eucs->capacity);
    }
    eucs->entries[eucs->count].u = u;
    eucs->entries[eucs->count].v = v;
    eucs->entries[eucs->count].twu = twu;
    eucs->count++;
}

static double eucs_get(EUCS *eucs, uint32_t u, uint32_t v) {
    if (u > v) { uint32_t t = u; u = v; v = t; }
    for (size_t i = 0; i < eucs->count; i++) {
        if (eucs->entries[i].u == u && eucs->entries[i].v == v) return eucs->entries[i].twu;
    }
    return 0;
}

/* --- UTILITY LIST JOIN --- */

static UtilityList* construct(UtilityList *p, UtilityList *px, UtilityList *py) {
    UtilityList *pxy = malloc(sizeof(UtilityList));
    pxy->count = px->count + 1;
    pxy->items = malloc(sizeof(uint32_t) * pxy->count);
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->count);
    pxy->items[px->count] = py->items[py->count - 1];
    
    pxy->tuples = malloc(sizeof(UtilityTuple) * (px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count));
    pxy->tuple_count = 0;
    pxy->sum_iputil = 0;
    pxy->sum_inutil = 0;
    pxy->sum_rutil = 0;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iputil = px->tuples[ix].iputil + py->tuples[iy].iputil;
            double inutil = px->tuples[ix].inutil + py->tuples[iy].inutil;
            
            if (p != NULL) {
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    iputil -= p->tuples[ip].iputil;
                    inutil -= p->tuples[ip].inutil;
                }
            }
            
            pxy->tuples[pxy->tuple_count].tid = tid;
            pxy->tuples[pxy->tuple_count].iputil = iputil;
            pxy->tuples[pxy->tuple_count].inutil = inutil;
            pxy->tuples[pxy->tuple_count].rutil = py->tuples[iy].rutil;
            
            pxy->sum_iputil += iputil;
            pxy->sum_inutil += inutil;
            pxy->sum_rutil += py->tuples[iy].rutil;
            
            pxy->tuple_count++;
            ix++; iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) ix++;
        else iy++;
    }
    return pxy;
}

static void free_utility_list(UtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    free(ul);
}

/* --- RECURSIVE SEARCH --- */

static size_t total_hui_count = 0;
static size_t total_items_sum = 0;

static void search(UtilityList *p, UtilityList **extensions, size_t ext_count, double min_util, EUCS *eucs, bool *is_positive) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *px = extensions[i];
        
        // HUI Check: u(X) = sum_iputil + sum_inutil
        if (px->sum_iputil + px->sum_inutil >= min_util) {
            total_hui_count++;
            total_items_sum += px->count;
        }

        uint32_t last_item_x = px->items[px->count - 1];
        bool x_is_positive = is_positive[last_item_x];

        // Pruning Conditions
        bool can_extend_positive = (px->sum_iputil + px->sum_rutil >= min_util);
        bool can_extend_negative = (px->sum_iputil >= min_util);

        if (can_extend_positive || can_extend_negative) {
            UtilityList **ext_px = malloc(sizeof(UtilityList*) * (ext_count - i - 1));
            size_t ext_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *py = extensions[j];
                uint32_t last_item_y = py->items[py->count - 1];
                bool y_is_positive = is_positive[last_item_y];

                if (y_is_positive) {
                    if (can_extend_positive) {
                        // For positive extensions, use EUCS and (sum_iputil + sum_rutil)
                        if (eucs_get(eucs, last_item_x, last_item_y) >= min_util) {
                            ext_px[ext_px_count++] = construct(p, px, py);
                        }
                    }
                } else {
                    if (can_extend_negative) {
                        // For negative extensions, use Property 11 (sum_iputil)
                        // Note: EUCS is not used for negative items
                        ext_px[ext_px_count++] = construct(p, px, py);
                    }
                }
            }

            if (ext_px_count > 0) {
                search(px, ext_px, ext_px_count, min_util, eucs, is_positive);
            }

            for (size_t j = 0; j < ext_px_count; j++) free_utility_list(ext_px[j]);
            free(ext_px);
        }
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_FHN_Params *p = (DM_FHN_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    total_hui_count = 0;
    total_items_sum = 0;

    printf("[FHN] Starting Phase 1 (TWU and Ordering)...\n");

    // 1. Determine which items are positive
    bool *is_positive = calloc(ds->max_id + 1, sizeof(bool));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > 0) is_positive[data[i].items[j].id] = true;
        }
    }

    // 2. Calculate Redefined TU (sum of positive items)
    double *redefined_tu = calloc(ds->count, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            if (is_positive[data[i].items[j].id]) {
                redefined_tu[i] += data[i].items[j].utility;
            }
        }
    }

    // 3. Calculate Redefined TWU
    double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            twu_counts[data[i].items[j].id] += redefined_tu[i];
        }
    }

    // 4. Filter and Sort Items
    ItemInfo *items = malloc(sizeof(ItemInfo) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu_counts[i] >= min_util) {
            items[item_count].id = i;
            items[item_count].twu = twu_counts[i];
            items[item_count].is_positive = is_positive[i];
            item_count++;
        }
    }
    qsort(items, item_count, sizeof(ItemInfo), cmp_item_info);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) rank[items[i].id] = i;

    printf("[FHN] Starting Phase 2 (Building Structures)...\n");

    UtilityList **initial_ext = malloc(sizeof(UtilityList*) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        initial_ext[i] = malloc(sizeof(UtilityList));
        initial_ext[i]->items = malloc(sizeof(uint32_t));
        initial_ext[i]->items[0] = items[i].id;
        initial_ext[i]->count = 1;
        initial_ext[i]->tuples = malloc(sizeof(UtilityTuple) * 16);
        initial_ext[i]->tuple_count = 0;
        initial_ext[i]->sum_iputil = 0;
        initial_ext[i]->sum_inutil = 0;
        initial_ext[i]->sum_rutil = 0;
    }

    EUCS eucs = { malloc(sizeof(EUCS_Entry) * 1024), 0, 1024 };

    for (size_t i = 0; i < ds->count; i++) {
        // Filter and sort items in transaction by rank
        uint32_t *t_items = malloc(sizeof(uint32_t) * data[i].count);
        double *t_utils = malloc(sizeof(double) * data[i].count);
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j].id;
            if (rank[id] != 0xFFFFFFFF) {
                t_items[t_count] = id;
                t_utils[t_count] = data[i].items[j].utility;
                t_count++;
            }
        }
        
        // Sort by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t temp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = temp_i;
                    double temp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = temp_u;
                }
            }
        }

        double remaining_positive_utility = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t item_id = t_items[j];
            uint32_t r = rank[item_id];
            UtilityList *ul = initial_ext[r];
            
            if (ul->tuple_count % 16 == 0 && ul->tuple_count > 0) {
                ul->tuples = realloc(ul->tuples, sizeof(UtilityTuple) * (ul->tuple_count + 16));
            }
            
            ul->tuples[ul->tuple_count].tid = i;
            if (is_positive[item_id]) {
                ul->tuples[ul->tuple_count].iputil = t_utils[j];
                ul->tuples[ul->tuple_count].inutil = 0;
                ul->sum_iputil += t_utils[j];
            } else {
                ul->tuples[ul->tuple_count].iputil = 0;
                ul->tuples[ul->tuple_count].inutil = t_utils[j];
                ul->sum_inutil += t_utils[j];
            }
            ul->tuples[ul->tuple_count].rutil = remaining_positive_utility;
            ul->sum_rutil += remaining_positive_utility;
            ul->tuple_count++;

            // Update EUCS (only for positive item pairs)
            if (is_positive[item_id]) {
                for (size_t k = 0; k < j; k++) {
                    if (is_positive[t_items[k]]) {
                        eucs_add(&eucs, t_items[j], t_items[k], redefined_tu[i]);
                    }
                }
                remaining_positive_utility += t_utils[j];
            }
        }
        free(t_items); free(t_utils);
    }

    printf("[FHN] Starting Recursive Search...\n");
    search(NULL, initial_ext, item_count, min_util, &eucs, is_positive);

    printf("[FHN] Found %zu High Utility Itemsets.\n", total_hui_count);

    // Cleanup
    for (size_t i = 0; i < item_count; i++) free_utility_list(initial_ext[i]);
    free(initial_ext);
    free(items); free(rank); free(twu_counts);
    free(eucs.entries);
    free(is_positive); free(redefined_tu);

    dm_bench_record_results(total_hui_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm fhn_algo = {
    .id = "fhn",
    .name = "FHN Algorithm",
    .description = "Faster High-Utility Itemset Mining with Negative Unit Profits.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(fhn_algo)
