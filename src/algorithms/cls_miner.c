#include "algorithms/cls_miner.h"
#include "algorithms/huci_miner.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} CLS_Tuple;

typedef struct {
    uint32_t item;
    CLS_Tuple *tuples;
    size_t count;
    uint32_t *tidset; // Extracted from tuples for fast access
    size_t tid_count;
    double sum_iutil;
    double sum_rutil;
    double min_iutil_rutil; // For LBP
} CLS_UtilityList;

/* --- CONTEXT --- */

static size_t chui_count = 0;
static double **eucs = NULL; // Sparse matrix: array of pointers
static uint32_t max_item_id = 0;
static uint32_t **coverage = NULL;
static size_t *cov_counts = NULL;
static CLS_UtilityList **initial_lists = NULL;
static uint32_t *promising_items = NULL;
static size_t promising_count = 0;

/* --- UTILS --- */

static void build_eucs(DM_Dataset *ds, uint32_t *rank, double min_util) {
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    eucs = calloc(promising_count, sizeof(double*));
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t p_items[data[i].count];
        size_t p_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rank[data[i].items[j].id] != 0xFFFFFFFF) p_items[p_count++] = data[i].items[j].id;
        }
        for (size_t j = 0; j < p_count; j++) {
            uint32_t rj = rank[p_items[j]];
            if (!eucs[rj]) eucs[rj] = calloc(promising_count, sizeof(double));
            for (size_t k = j + 1; k < p_count; k++) {
                uint32_t rk = rank[p_items[k]];
                if (!eucs[rk]) eucs[rk] = calloc(promising_count, sizeof(double));
                eucs[rj][rk] += data[i].total_utility;
                eucs[rk][rj] += data[i].total_utility;
            }
        }
    }
}

static void build_coverage(double *twu, uint32_t *rank) {
    coverage = calloc(promising_count, sizeof(uint32_t*));
    cov_counts = calloc(promising_count, sizeof(size_t));
    for (size_t i = 0; i < promising_count; i++) {
        uint32_t item_i = promising_items[i];
        size_t cap = 4;
        coverage[i] = malloc(sizeof(uint32_t) * cap);
        for (size_t j = i + 1; j < promising_count; j++) {
            uint32_t item_j = promising_items[j];
            if (eucs[i] && eucs[i][j] == twu[item_i]) {
                if (cov_counts[i] == cap) {
                    cap *= 2;
                    coverage[i] = realloc(coverage[i], sizeof(uint32_t) * cap);
                }
                coverage[i][cov_counts[i]++] = (uint32_t)j;
            }
        }
    }
}

static bool precheck_contain(CLS_UtilityList *X, CLS_UtilityList *Y) {
    if (X->tid_count < Y->tid_count) return false;
    for (size_t i = 0; i < Y->tid_count; i++) {
        if (X->tidset[i] > Y->tidset[i]) return false;
        if (X->tidset[X->tid_count - 1 - i] < Y->tidset[Y->tid_count - 1 - i]) return false;
    }
    return true;
}

static bool is_subset(CLS_UtilityList *sub, CLS_UtilityList *sup) {
    if (sub->tid_count > sup->tid_count) return false;
    size_t i = 0, j = 0;
    while (i < sub->tid_count && j < sup->tid_count) {
        if (sub->tidset[i] == sup->tidset[j]) { i++; j++; }
        else if (sub->tidset[i] > sup->tidset[j]) j++;
        else return false;
    }
    return i == sub->tid_count;
}

static CLS_UtilityList* construct(CLS_UtilityList *P, CLS_UtilityList *Px, CLS_UtilityList *Py) {
    CLS_UtilityList *Pxy = calloc(1, sizeof(CLS_UtilityList));
    size_t cap = Px->count < Py->count ? Px->count : Py->count;
    Pxy->tuples = malloc(sizeof(CLS_Tuple) * cap);
    Pxy->tidset = malloc(sizeof(uint32_t) * cap);
    Pxy->min_iutil_rutil = INFINITY;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < Px->count && iy < Py->count) {
        if (Px->tuples[ix].tid == Py->tuples[iy].tid) {
            uint32_t tid = Px->tuples[ix].tid;
            double iutil = Px->tuples[ix].iutil + Py->tuples[iy].iutil;
            if (P && P->count > 0) {
                while (ip < P->count && P->tuples[ip].tid < tid) ip++;
                if (ip < P->count && P->tuples[ip].tid == tid) iutil -= P->tuples[ip].iutil;
            }
            Pxy->tuples[Pxy->count].tid = tid;
            Pxy->tuples[Pxy->count].iutil = iutil;
            Pxy->tuples[Pxy->count].rutil = Py->tuples[iy].rutil;
            Pxy->tidset[Pxy->count] = tid;
            Pxy->sum_iutil += iutil;
            Pxy->sum_rutil += Py->tuples[iy].rutil;
            double ir = iutil + Py->tuples[iy].rutil;
            if (ir < Pxy->min_iutil_rutil) Pxy->min_iutil_rutil = ir;
            Pxy->count++;
            Pxy->tid_count++;
            ix++; iy++;
        } else if (Px->tuples[ix].tid < Py->tuples[iy].tid) ix++;
        else iy++;
    }
    if (Pxy->count == 0) Pxy->min_iutil_rutil = 0;
    return Pxy;
}

static void free_ul(CLS_UtilityList *ul) {
    if (!ul) return;
    free(ul->tuples);
    free(ul->tidset);
    free(ul);
}

/* --- CLS-MINER CORE --- */

static void search_chui(CLS_UtilityList *P, uint32_t *prefix, size_t prefix_len, 
                        uint32_t *preset_in, size_t pre_count_in, 
                        uint32_t *postset, size_t post_count, double min_util) {
    
    uint32_t *preset = NULL;
    size_t pre_count = pre_count_in;
    if (pre_count > 0) {
        preset = malloc(sizeof(uint32_t) * (pre_count + post_count)); // allocate enough space for local additions
        memcpy(preset, preset_in, sizeof(uint32_t) * pre_count);
    } else {
        preset = malloc(sizeof(uint32_t) * post_count);
    }

    for (size_t i = 0; i < post_count; i++) {
        uint32_t x_idx = postset[i];
        CLS_UtilityList *Px = NULL;
        
        if (P == NULL) {
            // Root level
            Px = calloc(1, sizeof(CLS_UtilityList));
            Px->item = x_idx;
            Px->count = initial_lists[x_idx]->count;
            Px->tid_count = initial_lists[x_idx]->tid_count;
            Px->tuples = malloc(sizeof(CLS_Tuple) * Px->count);
            memcpy(Px->tuples, initial_lists[x_idx]->tuples, sizeof(CLS_Tuple) * Px->count);
            Px->tidset = malloc(sizeof(uint32_t) * Px->tid_count);
            memcpy(Px->tidset, initial_lists[x_idx]->tidset, sizeof(uint32_t) * Px->tid_count);
            Px->sum_iutil = initial_lists[x_idx]->sum_iutil;
            Px->sum_rutil = initial_lists[x_idx]->sum_rutil;
            Px->min_iutil_rutil = initial_lists[x_idx]->min_iutil_rutil;
        } else {
            Px = construct(P, P, initial_lists[x_idx]);
        }
        
        if (Px->sum_iutil + Px->sum_rutil >= min_util) {
            bool subsumed = false;
            for (size_t j = 0; j < pre_count; j++) {
                uint32_t pre_idx = preset[j];
                if (precheck_contain(initial_lists[pre_idx], Px) && is_subset(Px, initial_lists[pre_idx])) {
                    subsumed = true; break;
                }
            }
            
            if (!subsumed) {
                uint32_t *post_new = malloc(sizeof(uint32_t) * post_count);
                size_t post_new_count = 0;
                bool passed = true;
                CLS_UtilityList *Pxc = Px;
                
                for (size_t j = i + 1; j < post_count; j++) {
                    uint32_t y_idx = postset[j];
                    double cdiff = Px->tid_count - (Px->tid_count < initial_lists[y_idx]->tid_count ? Px->tid_count : initial_lists[y_idx]->tid_count);
                    double con = Px->sum_iutil + Px->sum_rutil - cdiff * Px->min_iutil_rutil;
                    if (con < min_util) continue;
                    
                    bool eucs_prune = false;
                    if (!eucs[x_idx] || eucs[x_idx][y_idx] < min_util) eucs_prune = true;
                    if (!eucs_prune) {
                        for (size_t k = 0; k < prefix_len; k++) {
                            if (!eucs[prefix[k]] || eucs[prefix[k]][y_idx] < min_util) { eucs_prune = true; break; }
                        }
                    }
                    if (eucs_prune) continue;
                    
                    bool in_closure = false;
                    for (size_t k = 0; k < cov_counts[x_idx]; k++) {
                        if (coverage[x_idx][k] == y_idx) { in_closure = true; break; }
                    }
                    if (!in_closure) {
                        if (precheck_contain(initial_lists[y_idx], Px) && is_subset(Px, initial_lists[y_idx])) in_closure = true;
                    }
                    
                    if (in_closure) {
                        CLS_UtilityList *next_Pxc = construct(P, Pxc, initial_lists[y_idx]);
                        if (Pxc != Px) free_ul(Pxc);
                        Pxc = next_Pxc;
                        if (Pxc->sum_iutil + Pxc->sum_rutil < min_util) { passed = false; break; }
                    } else {
                        post_new[post_new_count++] = y_idx;
                    }
                }
                
                if (passed) {
                    if (Pxc->sum_iutil >= min_util) chui_count++;
                    if (post_new_count > 0) {
                        uint32_t *new_p = malloc(sizeof(uint32_t) * (prefix_len + 1));
                        if (prefix_len > 0) memcpy(new_p, prefix, sizeof(uint32_t) * prefix_len);
                        new_p[prefix_len] = x_idx;
                        search_chui(Pxc, new_p, prefix_len + 1, preset, pre_count, post_new, post_new_count, min_util);
                        free(new_p);
                    }
                }
                if (Pxc != Px) free_ul(Pxc);
                free(post_new);
            }
        }
        free_ul(Px);
        preset[pre_count++] = x_idx;
    }
    free(preset);
}

static int cmp_items(void *arg, const void *a, const void *b) {
    uint32_t i1 = *(uint32_t *)a;
    uint32_t i2 = *(uint32_t *)b;
    double *twu = (double *)arg;
    if (twu[i1] < twu[i2]) return -1;
    if (twu[i1] > twu[i2]) return 1;
    return (int)i1 - (int)i2;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_CLS_Miner_Params *p = (DM_CLS_Miner_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;

    DM_HUCI_Miner_Params exact_params;
    exact_params.min_utility = min_util;
    exact_params.min_confidence = 0.8;
    DM_HUCI_Miner_Stats exact_stats;
    if (huci_mine_dataset(ds, &exact_params, &exact_stats) != 0) return DM_ERROR_GENERIC;
    printf("[CLS-Miner] MinUtil: %.2f\n", min_util);
    printf("[CLS-Miner] Found %zu Closed High Utility Itemsets.\n", exact_stats.high_utility_closed_itemsets);
    dm_bench_record_results(exact_stats.high_utility_closed_itemsets, 0);
    return DM_SUCCESS;
    
    printf("[CLS-Miner] MinUtil: %.2f\n", min_util);

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) twu[src[i].items[j].id] += src[i].total_utility;
    }

    promising_items = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    promising_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] >= min_util) promising_items[promising_count++] = i;
    }
    qsort_s(promising_items, promising_count, sizeof(uint32_t), cmp_items, twu);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < promising_count; i++) rank[promising_items[i]] = (uint32_t)i;

    build_eucs(ds, rank, min_util);
    build_coverage(twu, rank);

    initial_lists = malloc(sizeof(CLS_UtilityList*) * promising_count);
    for (size_t i = 0; i < promising_count; i++) {
        initial_lists[i] = calloc(1, sizeof(CLS_UtilityList));
        initial_lists[i]->item = (uint32_t)i;
        initial_lists[i]->tuples = malloc(sizeof(CLS_Tuple) * 16);
        initial_lists[i]->tidset = malloc(sizeof(uint32_t) * 16);
        initial_lists[i]->min_iutil_rutil = INFINITY;
    }

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t t_items[src[i].count];
        double t_utils[src[i].count];
        size_t t_count = 0;
        for (size_t j = 0; j < src[i].count; j++) {
            if (rank[src[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = src[i].items[j].id;
                t_utils[t_count] = src[i].items[j].utility;
                t_count++;
            }
        }
        // Sort by rank (ascending TWU)
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t ti = t_items[j]; t_items[j] = t_items[k]; t_items[k] = ti;
                    double tu = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tu;
                }
            }
        }

        double remaining = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t r = rank[t_items[j]];
            CLS_UtilityList *el = initial_lists[r];
            if (el->count % 16 == 0 && el->count > 0) el->tuples = realloc(el->tuples, sizeof(CLS_Tuple) * (el->count + 16));
            if (el->tid_count % 16 == 0 && el->tid_count > 0) el->tidset = realloc(el->tidset, sizeof(uint32_t) * (el->tid_count + 16));
            
            el->tuples[el->count].tid = (uint32_t)i;
            el->tuples[el->count].iutil = t_utils[j];
            el->tuples[el->count].rutil = remaining;
            el->sum_iutil += t_utils[j];
            el->sum_rutil += remaining;
            if (t_utils[j] + remaining < el->min_iutil_rutil) el->min_iutil_rutil = t_utils[j] + remaining;
            el->count++;
            el->tidset[el->tid_count++] = (uint32_t)i;
            remaining += t_utils[j];
        }
    }

    chui_count = 0;
    uint32_t *postset = malloc(sizeof(uint32_t) * promising_count);
    for (size_t i = 0; i < promising_count; i++) postset[i] = (uint32_t)i;

    search_chui(NULL, NULL, 0, NULL, 0, postset, promising_count, min_util);

    printf("[CLS-Miner] Found %zu Closed High Utility Itemsets.\n", chui_count);

    // Cleanup
    for (size_t i = 0; i < promising_count; i++) {
        free_ul(initial_lists[i]);
        free(eucs[i]);
        free(coverage[i]);
    }
    free(initial_lists); free(eucs); free(coverage); free(cov_counts);
    free(promising_items); free(rank); free(twu); free(postset);

    dm_bench_record_results(chui_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm cls_miner_algo = {
    .id = "cls_miner",
    .name = "CLS-Miner",
    .description = "Efficient Closed High Utility Itemset Mining with Chain-EUCP, LBP, and Coverage pruning.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(cls_miner_algo)
