/**
 * @file ocean.c
 * @brief OCEAN: Fast Discovery of High Utility Occupancy Itemsets
 *
 * Reference:
 *   Bilong Shen, Zhaoduo Wen, Ying Zhao, Dongliang Zhou, and Weimin Zheng,
 *   "OCEAN: Fast Discovery of High Utility Occupancy Itemsets,"
 *   in Proc. 21st International Conference on Database Systems for Advanced Applications (DASFAA 2016),
 *   Springer LNCS 9642, 2016, pp. 355-369.
 *   DOI: 10.1007/978-3-319-32025-0_22
 */

#include "algorithms/ocean.h"
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
} OceanTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    OceanTuple *tuples;
    size_t tuple_count;
    double sum_iutil;
    double sum_rutil;
} OceanUtilityList;

typedef struct {
    uint32_t id;
    double twu;
    size_t support;
} ItemTWU;

/* --- UTILS --- */

static uint32_t *rank = NULL;

static int cmp_item_twu_asc(const void *a, const void *b) {
    const ItemTWU *ia = (const ItemTWU *)a;
    const ItemTWU *ib = (const ItemTWU *)b;
    if (ia->twu < ib->twu) return -1;
    if (ia->twu > ib->twu) return 1;
    return (int)(ia->id - ib->id);
}

static int cmp_item_rank(const void *a, const void *b) {
    uint32_t id1 = ((const DM_Item *)a)->id;
    uint32_t id2 = ((const DM_Item *)b)->id;
    uint32_t r1 = rank[id1];
    uint32_t r2 = rank[id2];
    return (r1 < r2) ? -1 : ((r1 > r2) ? 1 : 0);
}

static int cmp_desc_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) ? -1 : ((da < db) ? 1 : 0);
}

/* --- UTILITY LIST OPERATIONS --- */

static OceanUtilityList* construct(const OceanUtilityList *p, const OceanUtilityList *px, const OceanUtilityList *py) {
    OceanUtilityList *pxy = (OceanUtilityList *)malloc(sizeof(OceanUtilityList));
    if (!pxy) return NULL;
    pxy->count = px->count + 1;
    pxy->items = (uint32_t *)malloc(sizeof(uint32_t) * pxy->count);
    if (!pxy->items) { free(pxy); return NULL; }
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->count);
    pxy->items[px->count] = py->items[py->count - 1];

    size_t capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = (OceanTuple *)malloc(sizeof(OceanTuple) * (capacity ? capacity : 1));
    if (!pxy->tuples) { free(pxy->items); free(pxy); return NULL; }
    pxy->tuple_count = 0;
    pxy->sum_iutil = 0.0;
    pxy->sum_rutil = 0.0;

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
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) {
            ix++;
        } else {
            iy++;
        }
    }
    return pxy;
}

static void free_utility_list(OceanUtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    free(ul);
}

/* --- UPPER BOUND AND OCCUPANCY CALCULATION --- */

static double calculate_occu(const OceanUtilityList *px, const double *trans_utility) {
    if (px->tuple_count == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < px->tuple_count; i++) {
        uint32_t tid = px->tuples[i].tid;
        double tu = trans_utility[tid];
        if (tu > 0.0) {
            sum += px->tuples[i].iutil / tu;
        }
    }
    return sum / (double)px->tuple_count;
}

static double upper_bound(const OceanUtilityList *px, const double *trans_utility, size_t min_supp_count) {
    if (px->tuple_count < min_supp_count || min_supp_count == 0) return 0.0;
    double *v = (double *)malloc(sizeof(double) * px->tuple_count);
    if (!v) return 0.0;
    for (size_t i = 0; i < px->tuple_count; i++) {
        uint32_t tid = px->tuples[i].tid;
        double tu = trans_utility[tid];
        v[i] = (tu > 0.0) ? (px->tuples[i].iutil + px->tuples[i].rutil) / tu : 0.0;
    }
    qsort(v, px->tuple_count, sizeof(double), cmp_desc_double);
    double sum = 0.0;
    for (size_t i = 0; i < min_supp_count; i++) {
        sum += v[i];
    }
    free(v);
    return sum / (double)min_supp_count;
}

/* --- RECURSIVE SEARCH (Algorithm 3: Prune) --- */

static size_t total_huoi_count = 0;
static size_t total_items_sum = 0;

static DM_PatternCallback g_ocean_cb = NULL;
static void *g_ocean_cb_data = NULL;
static size_t ocean_visited_nodes = 0;

size_t ocean_get_visited_nodes(void) {
    return ocean_visited_nodes;
}

static void ocean_prune(OceanUtilityList *p, OceanUtilityList **exp, size_t exp_count,
                        double min_occu, size_t min_supp_count, const double *trans_utility) {
    for (size_t i = 0; i < exp_count; i++) {
        ocean_visited_nodes++;
        OceanUtilityList *px = exp[i];

        /* Line 2: if Occu(PX.UL) >= minOccu and Size(PX.UL) >= minSupp * |D| then Output PX */
        if (px->tuple_count >= min_supp_count) {
            double occu = calculate_occu(px, trans_utility);
            if (occu >= min_occu) {
                total_huoi_count++;
                total_items_sum += px->count;
                if (g_ocean_cb) {
                    g_ocean_cb(px->items, px->count, px->tuple_count, px->sum_iutil, occu, g_ocean_cb_data);
                }
            }
        }

        /* Line 5: maxOccu = UpperBound(PX) */
        double max_occu = upper_bound(px, trans_utility, min_supp_count);

        /* Line 6: if maxOccu >= minOccu and Size(PX.UL) >= minSupp * |D| then */
        if (max_occu >= min_occu && px->tuple_count >= min_supp_count) {
            OceanUtilityList **ex_px = (OceanUtilityList **)malloc(sizeof(OceanUtilityList *) * (exp_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < exp_count; j++) {
                OceanUtilityList *py = exp[j];
                OceanUtilityList *pxy = construct(p, px, py);
                if (pxy && pxy->tuple_count >= min_supp_count) {
                    ex_px[ex_px_count++] = pxy;
                } else if (pxy) {
                    free_utility_list(pxy);
                }
            }

            if (ex_px_count > 0) {
                ocean_prune(px, ex_px, ex_px_count, min_occu, min_supp_count, trans_utility);
            }

            for (size_t j = 0; j < ex_px_count; j++) {
                free_utility_list(ex_px[j]);
            }
            free(ex_px);
        }
    }
}

/* --- TOP-LEVEL EXECUTION (Algorithm 2: OCEAN) --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_OCEAN_Params *p = (DM_OCEAN_Params *)params;
    double min_occu = p ? p->min_occupancy : 0.5;
    double min_supp_arg = p ? p->min_support : 0.01;
    g_ocean_cb = p ? p->on_pattern : NULL;
    g_ocean_cb_data = p ? p->user_data : NULL;

    size_t min_supp_count = 1;
    if (min_supp_arg > 0.0 && min_supp_arg < 1.0) {
        min_supp_count = (size_t)ceil(min_supp_arg * (double)ds->count);
    } else if (min_supp_arg >= 1.0) {
        min_supp_count = (size_t)min_supp_arg;
    }
    if (min_supp_count < 1) min_supp_count = 1;

    DM_Trans_Utility *src_data = (DM_Trans_Utility *)ds->payload;
    total_huoi_count = 0;
    total_items_sum = 0;

    /* Build trans_utility lookup table */
    double *trans_utility = (double *)malloc(sizeof(double) * ds->count);
    if (!trans_utility) return DM_ERROR_MEMORY;
    for (size_t i = 0; i < ds->count; i++) {
        double tu = src_data[i].total_utility;
        if (tu <= 0.0) {
            for (size_t j = 0; j < src_data[i].count; j++) tu += src_data[i].items[j].utility;
        }
        trans_utility[i] = tu;
    }

    /* Compute TWU and Support per item */
    double *twu_counts = (double *)calloc(ds->max_id + 1, sizeof(double));
    size_t *supp_counts = (size_t *)calloc(ds->max_id + 1, sizeof(size_t));
    if (!twu_counts || !supp_counts) {
        free(trans_utility);
        free(twu_counts);
        free(supp_counts);
        return DM_ERROR_MEMORY;
    }

    for (size_t i = 0; i < ds->count; i++) {
        double tu = trans_utility[i];
        for (size_t j = 0; j < src_data[i].count; j++) {
            uint32_t id = src_data[i].items[j].id;
            twu_counts[id] += tu;
            supp_counts[id]++;
        }
    }

    /* Filter items by support threshold */
    ItemTWU *items = (ItemTWU *)malloc(sizeof(ItemTWU) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (supp_counts[i] >= min_supp_count) {
            items[item_count].id = i;
            items[item_count].twu = twu_counts[i];
            items[item_count].support = supp_counts[i];
            item_count++;
        }
    }

    /* Sort I in ascending order according to TWU (Algorithm 2, Line 2) */
    qsort(items, item_count, sizeof(ItemTWU), cmp_item_twu_asc);
    ocean_visited_nodes = item_count;

    rank = (uint32_t *)malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) {
        rank[items[i].id] = (uint32_t)i;
    }

    /* Sort transactions according to TWU ascending rank */
    for (size_t i = 0; i < ds->count; i++) {
        size_t valid_count = 0;
        for (size_t j = 0; j < src_data[i].count; j++) {
            if (rank[src_data[i].items[j].id] != 0xFFFFFFFF) {
                src_data[i].items[valid_count++] = src_data[i].items[j];
            }
        }
        src_data[i].count = valid_count;
        if (valid_count > 1) {
            qsort(src_data[i].items, valid_count, sizeof(DM_Item), cmp_item_rank);
        }
    }

    /* Build initial utility lists for singleton items */
    OceanUtilityList **ulists = (OceanUtilityList **)malloc(sizeof(OceanUtilityList *) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        ulists[i] = (OceanUtilityList *)malloc(sizeof(OceanUtilityList));
        ulists[i]->count = 1;
        ulists[i]->items = (uint32_t *)malloc(sizeof(uint32_t));
        ulists[i]->items[0] = items[i].id;
        ulists[i]->tuples = (OceanTuple *)malloc(sizeof(OceanTuple) * items[i].support);
        ulists[i]->tuple_count = 0;
        ulists[i]->sum_iutil = 0.0;
        ulists[i]->sum_rutil = 0.0;
    }

    for (size_t i = 0; i < ds->count; i++) {
        double rutil = 0.0;
        for (size_t j = src_data[i].count; j-- > 0; ) {
            uint32_t id = src_data[i].items[j].id;
            uint32_t r = rank[id];
            if (r != 0xFFFFFFFF) {
                size_t idx = ulists[r]->tuple_count++;
                ulists[r]->tuples[idx].tid = (uint32_t)i;
                ulists[r]->tuples[idx].iutil = src_data[i].items[j].utility;
                ulists[r]->tuples[idx].rutil = rutil;
                ulists[r]->sum_iutil += src_data[i].items[j].utility;
                ulists[r]->sum_rutil += rutil;
            }
            rutil += src_data[i].items[j].utility;
        }
    }

    /* Execute depth-first search with upper bound pruning */
    ocean_prune(NULL, ulists, item_count, min_occu, min_supp_count, trans_utility);

    printf("[OCEAN] Found %zu High Utility Occupancy Itemsets.\n", total_huoi_count);

    /* Free all memory */
    for (size_t i = 0; i < item_count; i++) {
        free_utility_list(ulists[i]);
    }
    free(ulists);
    free(items);
    free(rank);
    free(twu_counts);
    free(supp_counts);
    free(trans_utility);
    g_ocean_cb = NULL;
    g_ocean_cb_data = NULL;

    dm_bench_record_results(total_huoi_count, total_items_sum);
    return DM_SUCCESS;
}

DM_Algorithm ocean_algo = {
    .id = "ocean",
    .name = "OCEAN Algorithm",
    .description = "Fast Discovery of High Utility Occupancy Itemsets using Utility-Lists and Upper Bound Pruning (Shen et al. DASFAA 2016).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(ocean_algo)

