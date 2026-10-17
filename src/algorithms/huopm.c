/**
 * @file huopm.c
 * @brief HUOPM: High-Utility Occupancy Pattern Mining
 *
 * Reference:
 *   Wensheng Gan, Jerry Chun-Wei Lin, Philippe Fournier-Viger, Han-Chieh Chao, and Philip S. Yu,
 *   "HUOPM: High-Utility Occupancy Pattern Mining,"
 *   IEEE Transactions on Cybernetics, vol. 50, no. 3, pp. 1195-1208, March 2020.
 *   DOI: 10.1109/TCYB.2019.2894367
 */

#include "algorithms/huopm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/* --- DATA STRUCTURES --- */

/**
 * @brief UO-list tuple: <tid, uo, ruo>
 * uo  = u(X, t) / TU(t)
 * ruo = ru(X, t) / TU(t)
 */
typedef struct {
    uint32_t tid;
    double uo;
    double ruo;
} UOTuple;

/**
 * @brief FU-table summary: <sup, uo, ruo>
 */
typedef struct {
    size_t sup;
    double uo;   /**< Average utility occupancy: (1 / sup) * sum(uo_t) */
    double ruo;  /**< Average remaining utility occupancy: (1 / sup) * sum(ruo_t) */
} FUTable;

/**
 * @brief UO-list node representing an itemset X
 */
typedef struct {
    uint32_t *items;
    size_t count;
    UOTuple *tuples;
    size_t tuple_count;
    FUTable fut;
} UOList;

typedef struct {
    uint32_t id;
    size_t support;
} ItemSupport;

/* --- GLOBAL / THREAD-LOCAL HELPERS --- */

static uint32_t *rank_map = NULL;

static int cmp_item_supp_asc(const void *a, const void *b) {
    const ItemSupport *ia = (const ItemSupport *)a;
    const ItemSupport *ib = (const ItemSupport *)b;
    if (ia->support < ib->support) return -1;
    if (ia->support > ib->support) return 1;
    return (ia->id < ib->id) ? -1 : ((ia->id > ib->id) ? 1 : 0);
}

static int cmp_item_rank(const void *a, const void *b) {
    uint32_t id1 = ((const DM_Item *)a)->id;
    uint32_t id2 = ((const DM_Item *)b)->id;
    uint32_t r1 = rank_map[id1];
    uint32_t r2 = rank_map[id2];
    return (r1 < r2) ? -1 : ((r1 > r2) ? 1 : 0);
}

static int cmp_desc_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) ? -1 : ((da < db) ? 1 : 0);
}

static void free_uolist(UOList *ul) {
    if (ul) {
        if (ul->items) free(ul->items);
        if (ul->tuples) free(ul->tuples);
        free(ul);
    }
}

/* --- ALGORITHM 1: CONSTRUCTION WITH STRATEGY 3 --- */

static UOList* construct(const UOList *p, const UOList *px, const UOList *py, size_t min_supp_count) {
    size_t capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    if (capacity < min_supp_count) return NULL;

    UOList *pxy = (UOList *)malloc(sizeof(UOList));
    if (!pxy) return NULL;

    pxy->count = px->count + 1;
    pxy->items = (uint32_t *)malloc(sizeof(uint32_t) * pxy->count);
    if (!pxy->items) {
        free(pxy);
        return NULL;
    }
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->count);
    pxy->items[px->count] = py->items[py->count - 1];

    pxy->tuples = (UOTuple *)malloc(sizeof(UOTuple) * capacity);
    if (!pxy->tuples) {
        free(pxy->items);
        free(pxy);
        return NULL;
    }
    pxy->tuple_count = 0;

    double sum_uo = 0.0;
    double sum_ruo = 0.0;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        /* Strategy 3: Early construction abandonment */
        if (pxy->tuple_count + (px->tuple_count - ix) < min_supp_count) {
            free_uolist(pxy);
            return NULL;
        }
        if (pxy->tuple_count + (py->tuple_count - iy) < min_supp_count) {
            free_uolist(pxy);
            return NULL;
        }

        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double uo = px->tuples[ix].uo + py->tuples[iy].uo;
            if (p != NULL) {
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    uo -= p->tuples[ip].uo;
                }
            }
            double ruo = py->tuples[iy].ruo;

            pxy->tuples[pxy->tuple_count].tid = tid;
            pxy->tuples[pxy->tuple_count].uo = uo;
            pxy->tuples[pxy->tuple_count].ruo = ruo;
            pxy->tuple_count++;

            sum_uo += uo;
            sum_ruo += ruo;

            ix++;
            iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) {
            ix++;
        } else {
            iy++;
        }
    }

    if (pxy->tuple_count < min_supp_count) {
        free_uolist(pxy);
        return NULL;
    }

    pxy->fut.sup = pxy->tuple_count;
    pxy->fut.uo = sum_uo / (double)pxy->tuple_count;
    pxy->fut.ruo = sum_ruo / (double)pxy->tuple_count;
    return pxy;
}

/* --- ALGORITHM 4: UPPER BOUND WITH STRATEGY 2 --- */

static double upper_bound(const UOList *px, size_t min_supp_count) {
    if (px->tuple_count < min_supp_count || min_supp_count == 0) return 0.0;

    if (px->tuple_count == min_supp_count) {
        double sum = 0.0;
        for (size_t i = 0; i < px->tuple_count; i++) {
            sum += (px->tuples[i].uo + px->tuples[i].ruo);
        }
        return sum / (double)min_supp_count;
    }

    double *v = (double *)malloc(sizeof(double) * px->tuple_count);
    if (!v) return 0.0;

    for (size_t i = 0; i < px->tuple_count; i++) {
        v[i] = px->tuples[i].uo + px->tuples[i].ruo;
    }

    qsort(v, px->tuple_count, sizeof(double), cmp_desc_double);

    double sum = 0.0;
    for (size_t i = 0; i < min_supp_count; i++) {
        sum += v[i];
    }
    free(v);
    return sum / (double)min_supp_count;
}

/* --- ALGORITHM 3: HUOP-SEARCH --- */

static size_t total_huop_count = 0;
static size_t total_items_sum = 0;
static size_t huopm_visited_nodes = 0;
static DM_PatternCallback g_huopm_cb = NULL;
static void *g_huopm_cb_data = NULL;
static const double *g_huopm_trans_utility = NULL;

size_t huopm_get_visited_nodes(void) {
    return huopm_visited_nodes;
}

static void huop_search(const UOList *p, UOList **exten, size_t exten_count,
                        double min_occu, size_t min_supp_count) {
    for (size_t i = 0; i < exten_count; i++) {
        huopm_visited_nodes++;
        UOList *px = exten[i];

        /* Line 2-6: Check if px is a HUOP */
        if (px->fut.sup >= min_supp_count && px->fut.uo >= min_occu) {
            total_huop_count++;
            total_items_sum += px->count;
            if (g_huopm_cb) {
                double util = 0.0;
                if (g_huopm_trans_utility) {
                    for (size_t t = 0; t < px->tuple_count; t++) {
                        util += px->tuples[t].uo * g_huopm_trans_utility[px->tuples[t].tid];
                    }
                }
                g_huopm_cb(px->items, px->count, px->fut.sup, util, px->fut.uo, g_huopm_cb_data);
            }
        }

        /* Line 7-8: Strategy 2 - Upper bound check */
        double phi_hat = upper_bound(px, min_supp_count);
        if (phi_hat >= min_occu) {
            UOList **exten_px = (UOList **)malloc(sizeof(UOList *) * (exten_count - i - 1));
            size_t exten_px_count = 0;

            for (size_t j = i + 1; j < exten_count; j++) {
                UOList *py = exten[j];
                UOList *pxy = construct(p, px, py, min_supp_count);
                /* Strategy 4: Discard if NULL or infrequent */
                if (pxy != NULL && pxy->fut.sup >= min_supp_count) {
                    exten_px[exten_px_count++] = pxy;
                } else if (pxy != NULL) {
                    free_uolist(pxy);
                }
            }

            if (exten_px_count > 0) {
                huop_search(px, exten_px, exten_px_count, min_occu, min_supp_count);
            }

            for (size_t j = 0; j < exten_px_count; j++) {
                free_uolist(exten_px[j]);
            }
            free(exten_px);
        }
    }
}

/* --- ALGORITHM 2: HUOPM MAIN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_HUOPM_Params *p = (DM_HUOPM_Params *)params;
    double min_occu = p ? p->min_occupancy : 0.5;
    double min_supp_arg = p ? p->min_support : 0.01;
    g_huopm_cb = p ? p->on_pattern : NULL;
    g_huopm_cb_data = p ? p->user_data : NULL;

    size_t min_supp_count = 1;
    if (min_supp_arg > 0.0 && min_supp_arg < 1.0) {
        min_supp_count = (size_t)ceil(min_supp_arg * (double)ds->count);
    } else if (min_supp_arg >= 1.0) {
        min_supp_count = (size_t)min_supp_arg;
    }
    if (min_supp_count < 1) min_supp_count = 1;

    total_huop_count = 0;
    total_items_sum = 0;

    DM_Trans_Utility *src_data = (DM_Trans_Utility *)ds->payload;
    uint32_t max_id = ds->max_id;

    size_t *supp_counts = (size_t *)calloc(max_id + 1, sizeof(size_t));
    double *trans_utility = (double *)calloc(ds->count, sizeof(double));
    if (!supp_counts || !trans_utility) {
        free(supp_counts);
        free(trans_utility);
        return DM_ERROR_MEMORY;
    }
    g_huopm_trans_utility = trans_utility;

    for (size_t i = 0; i < ds->count; i++) {
        double tu = src_data[i].total_utility;
        if (tu <= 0.0) {
            for (size_t j = 0; j < src_data[i].count; j++) tu += src_data[i].items[j].utility;
        }
        trans_utility[i] = tu;

        for (size_t j = 0; j < src_data[i].count; j++) {
            supp_counts[src_data[i].items[j].id]++;
        }
    }

    /* Step 2: Find I* <- {i | sup(i) >= min_supp_count} */
    size_t frequent_count = 0;
    for (uint32_t i = 0; i <= max_id; i++) {
        if (supp_counts[i] >= min_supp_count) {
            frequent_count++;
        }
    }

    if (frequent_count == 0) {
        free(supp_counts);
        free(trans_utility);
        dm_bench_record_results(0, 0);
        return DM_SUCCESS;
    }

    ItemSupport *items = (ItemSupport *)malloc(sizeof(ItemSupport) * frequent_count);
    size_t idx = 0;
    for (uint32_t i = 0; i <= max_id; i++) {
        if (supp_counts[i] >= min_supp_count) {
            items[idx].id = i;
            items[idx].support = supp_counts[i];
            idx++;
        }
    }

    /* Step 3: Sort I* in support ascending order (HUOPM_supas) */
    qsort(items, frequent_count, sizeof(ItemSupport), cmp_item_supp_asc);
    huopm_visited_nodes = frequent_count;

    rank_map = (uint32_t *)malloc(sizeof(uint32_t) * (max_id + 1));
    for (uint32_t i = 0; i <= max_id; i++) {
        rank_map[i] = 0xFFFFFFFF;
    }
    for (size_t i = 0; i < frequent_count; i++) {
        rank_map[items[i].id] = (uint32_t)i;
    }

    /* Step 4: Scan D to build UO-list and FU-table for each 1-item */
    UOList **ulists = (UOList **)malloc(sizeof(UOList *) * frequent_count);
    double *sum_uo_arr = (double *)calloc(frequent_count, sizeof(double));
    double *sum_ruo_arr = (double *)calloc(frequent_count, sizeof(double));

    for (size_t i = 0; i < frequent_count; i++) {
        ulists[i] = (UOList *)malloc(sizeof(UOList));
        ulists[i]->count = 1;
        ulists[i]->items = (uint32_t *)malloc(sizeof(uint32_t));
        ulists[i]->items[0] = items[i].id;
        ulists[i]->tuples = (UOTuple *)malloc(sizeof(UOTuple) * items[i].support);
        ulists[i]->tuple_count = 0;
        ulists[i]->fut.sup = items[i].support;
    }

    DM_Item *reordered = (DM_Item *)malloc(sizeof(DM_Item) * (max_id + 1));

    for (size_t t = 0; t < ds->count; t++) {
        double tu = trans_utility[t];
        if (tu <= 0.0) continue;

        size_t m = 0;
        for (size_t j = 0; j < src_data[t].count; j++) {
            uint32_t id = src_data[t].items[j].id;
            if (rank_map[id] != 0xFFFFFFFF) {
                reordered[m++] = src_data[t].items[j];
            }
        }
        if (m == 0) continue;

        qsort(reordered, m, sizeof(DM_Item), cmp_item_rank);

        double ru = 0.0;
        for (size_t j = m; j-- > 0; ) {
            uint32_t id = reordered[j].id;
            uint32_t r = rank_map[id];

            double u = reordered[j].utility;
            double uo = u / tu;
            double ruo = ru / tu;

            size_t pos = ulists[r]->tuple_count++;
            ulists[r]->tuples[pos].tid = (uint32_t)t;
            ulists[r]->tuples[pos].uo = uo;
            ulists[r]->tuples[pos].ruo = ruo;

            sum_uo_arr[r] += uo;
            sum_ruo_arr[r] += ruo;

            ru += u;
        }
    }
    free(reordered);

    /* Finalize FU-tables for 1-itemsets */
    for (size_t i = 0; i < frequent_count; i++) {
        ulists[i]->fut.sup = ulists[i]->tuple_count;
        if (ulists[i]->tuple_count > 0) {
            ulists[i]->fut.uo = sum_uo_arr[i] / (double)ulists[i]->tuple_count;
            ulists[i]->fut.ruo = sum_ruo_arr[i] / (double)ulists[i]->tuple_count;
        } else {
            ulists[i]->fut.uo = 0.0;
            ulists[i]->fut.ruo = 0.0;
        }
    }
    free(sum_uo_arr);
    free(sum_ruo_arr);

    /* Step 5: Call HUOP-Search */
    huop_search(NULL, ulists, frequent_count, min_occu, min_supp_count);

    printf("[HUOPM] Found %zu High-Utility Occupancy Patterns.\n", total_huop_count);

    /* Free all memory */
    for (size_t i = 0; i < frequent_count; i++) {
        free_uolist(ulists[i]);
    }
    free(ulists);
    free(items);
    free(rank_map);
    rank_map = NULL;
    free(supp_counts);
    free(trans_utility);
    g_huopm_cb = NULL;
    g_huopm_cb_data = NULL;
    g_huopm_trans_utility = NULL;

    dm_bench_record_results(total_huop_count, total_items_sum);
    return DM_SUCCESS;
}

DM_Algorithm huopm_algo = {
    .id = "huopm",
    .name = "HUOPM Algorithm",
    .description = "High-Utility Occupancy Pattern Mining with compact UO-lists and upper bound pruning (Gan et al. IEEE TCYB 2020).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(huopm_algo)
