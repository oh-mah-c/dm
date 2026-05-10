#include "algorithms/fhm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

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

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

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

/* --- EUCS --- */

static void eucs_add(EUCS *eucs, uint32_t u, uint32_t v, double twu) {
    if (u > v) { uint32_t t = u; u = v; v = t; }
    // Check if already exists (very basic search for now, could be hash map)
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
    pxy->sum_iutil = 0;
    pxy->sum_rutil = 0;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            if (p != NULL) {
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

static void search(UtilityList *p, UtilityList **extensions, size_t ext_count, double min_util, EUCS *eucs) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *px = extensions[i];
        if (px->sum_iutil >= min_util) {
            total_hui_count++;
            total_items_sum += px->count;
        }

        if (px->sum_iutil + px->sum_rutil >= min_util) {
            UtilityList **ext_px = malloc(sizeof(UtilityList*) * (ext_count - i - 1));
            size_t ext_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *py = extensions[j];
                uint32_t x = px->items[px->count - 1];
                uint32_t y = py->items[py->count - 1];
                
                if (eucs_get(eucs, x, y) >= min_util) {
                    ext_px[ext_px_count++] = construct(p, px, py);
                }
            }

            if (ext_px_count > 0) {
                search(px, ext_px, ext_px_count, min_util, eucs);
            }

            for (size_t j = 0; j < ext_px_count; j++) free_utility_list(ext_px[j]);
            free(ext_px);
        }
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_FHM_Params *p = (DM_FHM_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    total_hui_count = 0;
    total_items_sum = 0;

    printf("[FHM] Starting Phase 1 (TWU and Ordering)...\n");

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

    // Map ID to rank
    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) rank[items[i].id] = i;

    printf("[FHM] Starting Phase 2 (Building Structures)...\n");

    UtilityList **initial_ext = malloc(sizeof(UtilityList*) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        initial_ext[i] = malloc(sizeof(UtilityList));
        initial_ext[i]->items = malloc(sizeof(uint32_t));
        initial_ext[i]->items[0] = items[i].id;
        initial_ext[i]->count = 1;
        initial_ext[i]->tuples = malloc(sizeof(UtilityTuple) * 16);
        initial_ext[i]->tuple_count = 0;
        initial_ext[i]->sum_iutil = 0;
        initial_ext[i]->sum_rutil = 0;
    }

    EUCS eucs = { malloc(sizeof(EUCS_Entry) * 1024), 0, 1024 };

    for (size_t i = 0; i < ds->count; i++) {
        // Filter and sort items in transaction by rank
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
        
        // Sort by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t temp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = temp_i;
                    double temp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = temp_u;
                }
            }
        }

        double remaining_utility = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t item_id = t_items[j];
            uint32_t r = rank[item_id];
            UtilityList *ul = initial_ext[r];
            
            // Realloc if needed
            if (ul->tuple_count % 16 == 0 && ul->tuple_count > 0) {
                ul->tuples = realloc(ul->tuples, sizeof(UtilityTuple) * (ul->tuple_count + 16));
            }
            
            ul->tuples[ul->tuple_count].tid = i;
            ul->tuples[ul->tuple_count].iutil = t_utils[j];
            ul->tuples[ul->tuple_count].rutil = remaining_utility;
            ul->sum_iutil += t_utils[j];
            ul->sum_rutil += remaining_utility;
            ul->tuple_count++;

            // Update EUCS
            for (size_t k = 0; k < j; k++) {
                eucs_add(&eucs, t_items[j], t_items[k], data[i].total_utility);
            }
            
            remaining_utility += t_utils[j];
        }
        free(t_items); free(t_utils);
    }

    printf("[FHM] Starting Recursive Search...\n");
    search(NULL, initial_ext, item_count, min_util, &eucs);

    printf("[FHM] Found %zu High Utility Itemsets.\n", total_hui_count);

    // Cleanup
    for (size_t i = 0; i < item_count; i++) free_utility_list(initial_ext[i]);
    free(initial_ext);
    free(items); free(rank); free(twu_counts);
    free(eucs.entries);

    dm_bench_record_results(total_hui_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fhm",
    .name = "FHM Algorithm",
    .description = "Faster High-Utility Itemset Mining using Estimated Utility Co-occurrence Pruning (EUCP).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
