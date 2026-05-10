#include "algorithms/hui_miner.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} UtilityTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    UtilityTuple *tuples;
    size_t tuple_count;
    double sum_iutil;
    double sum_rutil;
} UtilityList;

/* --- UTILS --- */

typedef struct {
    uint32_t id;
    double twu;
} ItemTWU;

static int cmp_item_twu(const void *a, const void *b) {
    double twu1 = ((ItemTWU*)a)->twu;
    double twu2 = ((ItemTWU*)b)->twu;
    if (twu1 < twu2) return -1;
    if (twu1 > twu2) return 1;
    return (int)(((ItemTWU*)a)->id - ((ItemTWU*)b)->id);
}

static uint32_t *rank = NULL;

static int cmp_item_rank(const void *a, const void *b) {
    uint32_t r1 = rank[*(uint32_t*)a];
    uint32_t r2 = rank[*(uint32_t*)b];
    return (r1 < r2) ? -1 : ((r1 > r2) ? 1 : 0);
}

/* --- UTILITY LIST CONSTRUCTION --- */

static UtilityList* construct(UtilityList *p, UtilityList *px, UtilityList *py) {
    UtilityList *pxy = malloc(sizeof(UtilityList));
    pxy->count = px->count + 1;
    pxy->items = malloc(sizeof(uint32_t) * pxy->count);
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->count);
    pxy->items[px->count] = py->items[py->count - 1];
    
    // Initial capacity based on smaller list
    size_t capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = malloc(sizeof(UtilityTuple) * capacity);
    pxy->tuple_count = 0;
    pxy->sum_iutil = 0;
    pxy->sum_rutil = 0;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            
            if (p != NULL) {
                // Relabeling optimization: in the paper, tid can be used to find index in P.
                // But if we don't relabel, we search.
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    iutil -= p->tuples[ip].iutil;
                }
            }
            
            pxy->tuples[pxy->tuple_count].tid = tid;
            pxy->tuples[pxy->tuple_count].iutil = iutil;
            pxy->tuples[pxy->tuple_count].rutil = py->tuples[iy].rutil;
            pxy->sum_iutil += iutil;
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

static void hui_miner_mine(UtilityList *p, UtilityList **extensions, size_t ext_count, double min_util) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *px = extensions[i];
        
        if (px->sum_iutil >= min_util) {
            total_hui_count++;
            total_items_sum += px->count;
        }

        if (px->sum_iutil + px->sum_rutil >= min_util) {
            UtilityList **ex_px = malloc(sizeof(UtilityList*) * (ext_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *py = extensions[j];
                ex_px[ex_px_count++] = construct(p, px, py);
            }

            if (ex_px_count > 0) {
                hui_miner_mine(px, ex_px, ex_px_count, min_util);
            }

            for (size_t j = 0; j < ex_px_count; j++) free_utility_list(ex_px[j]);
            free(ex_px);
        }
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_HUI_Miner_Params *p = (DM_HUI_Miner_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    total_hui_count = 0;
    total_items_sum = 0;

    printf("[HUI-Miner] Phase 1: Calculating TWU and filtering...\n");
    double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) twu_counts[data[i].items[j].id] += data[i].total_utility;
    }

    ItemTWU *items = malloc(sizeof(ItemTWU) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu_counts[i] >= min_util) {
            items[item_count].id = i;
            items[item_count].twu = twu_counts[i];
            item_count++;
        }
    }
    qsort(items, item_count, sizeof(ItemTWU), cmp_item_twu);

    rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) rank[items[i].id] = (uint32_t)i;

    printf("[HUI-Miner] Phase 2: Building initial utility-lists...\n");
    UtilityList **initial_ext = malloc(sizeof(UtilityList*) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        initial_ext[i] = malloc(sizeof(UtilityList));
        initial_ext[i]->items = malloc(sizeof(uint32_t));
        initial_ext[i]->items[0] = items[i].id;
        initial_ext[i]->count = 1;
        initial_ext[i]->tuples = malloc(sizeof(UtilityTuple) * 8); // dynamic realloc
        initial_ext[i]->tuple_count = 0;
        initial_ext[i]->sum_iutil = 0;
        initial_ext[i]->sum_rutil = 0;
    }

    for (size_t i = 0; i < ds->count; i++) {
        // Filter and sort items by rank
        uint32_t *t_items = malloc(sizeof(uint32_t) * data[i].count);
        double *t_utils = malloc(sizeof(double) * data[i].count);
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rank[data[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = data[i].items[j].id;
                t_utils[t_count] = data[i].items[j].utility;
                t_count++;
            }
        }
        
        // Sorting items in transaction by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t tmp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = tmp_i;
                    double tmp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tmp_u;
                }
            }
        }

        double remaining_utility = 0;
        for (int j = (int)t_count - 1; j >= 0; j--) {
            uint32_t r = rank[t_items[j]];
            UtilityList *ul = initial_ext[r];
            
            if (ul->tuple_count > 0 && ul->tuple_count % 8 == 0) {
                ul->tuples = realloc(ul->tuples, sizeof(UtilityTuple) * (ul->tuple_count + 8));
            }
            
            ul->tuples[ul->tuple_count].tid = (uint32_t)i;
            ul->tuples[ul->tuple_count].iutil = t_utils[j];
            ul->tuples[ul->tuple_count].rutil = remaining_utility;
            ul->sum_iutil += t_utils[j];
            ul->sum_rutil += remaining_utility;
            ul->tuple_count++;
            
            remaining_utility += t_utils[j];
        }
        free(t_items); free(t_utils);
    }

    printf("[HUI-Miner] Starting recursive mining...\n");
    hui_miner_mine(NULL, initial_ext, item_count, min_util);

    printf("[HUI-Miner] Found %zu High Utility Itemsets.\n", total_hui_count);

    // Cleanup
    for (size_t i = 0; i < item_count; i++) free_utility_list(initial_ext[i]);
    free(initial_ext);
    free(items); free(rank); free(twu_counts);

    dm_bench_record_results(total_hui_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "huiminer",
    .name = "HUI-Miner",
    .description = "High Utility Itemset Mining without candidate generation using utility-lists.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
