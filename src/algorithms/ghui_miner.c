#include "algorithms/ghui_miner.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include "algorithms/efim_closed.h" // For CHUI mining if needed
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

/* --- BITSET --- */

typedef struct {
    uint64_t *bits;
    size_t size;
} GHUI_BitSet;

static GHUI_BitSet* bs_create(size_t n) {
    GHUI_BitSet *bs = malloc(sizeof(GHUI_BitSet));
    bs->size = (n + 63) / 64;
    bs->bits = calloc(bs->size, sizeof(uint64_t));
    return bs;
}

static void bs_free(GHUI_BitSet *bs) {
    if (bs) free(bs->bits);
    free(bs);
}

static void bs_set(GHUI_BitSet *bs, size_t i) {
    bs->bits[i / 64] |= (1ULL << (i % 64));
}

static void bs_and(GHUI_BitSet *res, GHUI_BitSet *a, GHUI_BitSet *b) {
    for (size_t i = 0; i < res->size; i++) res->bits[i] = a->bits[i] & b->bits[i];
}

static bool bs_is_empty(GHUI_BitSet *bs) {
    for (size_t i = 0; i < bs->size; i++) if (bs->bits[i]) return false;
    return true;
}

/* --- UTILITY LIST --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} GHUI_Tuple;

typedef struct {
    uint32_t item;
    GHUI_Tuple *tuples;
    size_t count;
    double sum_iutil;
    double sum_rutil;
    GHUI_BitSet **crit; // crit[item_in_prefix]
} GHUI_UtilityList;

/* --- CONTEXT --- */

static size_t hug_count = 0;
static size_t lug_count = 0;
static double **eucs = NULL;
static uint32_t promising_count = 0;
static uint32_t *promising_items = NULL;
static GHUI_UtilityList **initial_lists = NULL;
static GHUI_BitSet **item_tidsets = NULL;
static size_t num_transactions = 0;

/* --- LOGIC --- */

static GHUI_UtilityList* construct(GHUI_UtilityList *P, GHUI_UtilityList *Px, GHUI_UtilityList *Py, size_t prefix_len) {
    GHUI_UtilityList *Pxy = calloc(1, sizeof(GHUI_UtilityList));
    size_t cap = Px->count < Py->count ? Px->count : Py->count;
    Pxy->tuples = malloc(sizeof(GHUI_Tuple) * cap);
    
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
            Pxy->tuples[Pxy->count].rutil = Px->tuples[ix].rutil - Py->tuples[iy].iutil;
            Pxy->sum_iutil += iutil;
            Pxy->sum_rutil += Pxy->tuples[Pxy->count].rutil;
            Pxy->count++;
            ix++; iy++;
        } else if (Px->tuples[ix].tid < Py->tuples[iy].tid) ix++;
        else iy++;
    }
    
    // Critical transactions for generator check
    if (Pxy->count > 0) {
        Pxy->crit = malloc(sizeof(GHUI_BitSet*) * (prefix_len + 2));
        // crit(Pxy, z) = crit(Px, z) AND g(y)
        for (size_t i = 0; i < prefix_len + 1; i++) {
            Pxy->crit[i] = bs_create(num_transactions);
            bs_and(Pxy->crit[i], Px->crit[i], item_tidsets[Py->item]);
        }
        // crit(Pxy, y) = g(Px) \ g(y) -> approximated or calculated
        // The paper says: crit(X U {b}, a) = crit(X, a) AND g(b)
        // For the new item y, we need crit(Pxy, y) = g(Px) \ g(y)
        Pxy->crit[prefix_len + 1] = bs_create(num_transactions);
        GHUI_BitSet *gPx = bs_create(num_transactions);
        for (size_t i = 0; i < Px->count; i++) bs_set(gPx, Px->tuples[i].tid);
        for (size_t i = 0; i < gPx->size; i++) {
            Pxy->crit[prefix_len + 1]->bits[i] = gPx->bits[i] & (~item_tidsets[Py->item]->bits[i]);
        }
        bs_free(gPx);
    }

    return Pxy;
}

static void free_ul(GHUI_UtilityList *ul, size_t prefix_len) {
    if (!ul) return;
    free(ul->tuples);
    if (ul->crit) {
        for (size_t i = 0; i < prefix_len + 2; i++) bs_free(ul->crit[i]);
        free(ul->crit);
    }
    free(ul);
}

static bool is_generator(GHUI_UtilityList *ul, size_t prefix_len) {
    if (!ul->crit) return true;
    for (size_t i = 0; i < prefix_len + 2; i++) {
        if (bs_is_empty(ul->crit[i])) return false;
    }
    return true;
}

static void search(GHUI_UtilityList *P, GHUI_UtilityList **extensions, size_t ext_count, size_t prefix_len, double min_util) {
    for (size_t i = 0; i < ext_count; i++) {
        GHUI_UtilityList *Px = extensions[i];
        GHUI_UtilityList **next_extensions = malloc(sizeof(GHUI_UtilityList*) * (ext_count - i - 1));
        size_t next_ext_count = 0;
        
        for (size_t j = i + 1; j < ext_count; j++) {
            GHUI_UtilityList *Py = extensions[j];
            
            // EUCP Pruning
            if (eucs[Px->item] && eucs[Px->item][Py->item] < min_util) continue;
            
            GHUI_UtilityList *Pxy = construct(P, Px, Py, prefix_len);
            
            if (Pxy->count > 0) {
                // Generator Check
                if (Pxy->count != Px->count && Pxy->count != Py->count && is_generator(Pxy, prefix_len)) {
                    if (Pxy->sum_iutil + Pxy->sum_rutil >= min_util) {
                        if (Pxy->sum_iutil >= min_util) hug_count++;
                        next_extensions[next_ext_count++] = Pxy;
                    } else {
                        free_ul(Pxy, prefix_len);
                    }
                } else {
                    free_ul(Pxy, prefix_len);
                }
            } else {
                free_ul(Pxy, prefix_len);
            }
        }
        
        if (next_ext_count > 0) {
            search(Px, next_extensions, next_ext_count, prefix_len + 1, min_util);
        }
        
        for (size_t k = 0; k < next_ext_count; k++) free_ul(next_extensions[k], prefix_len);
        free(next_extensions);
    }
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
    DM_GHUI_Miner_Params *p = (DM_GHUI_Miner_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    printf("[HUG-Miner] MinUtil: %.2f\n", min_util);
    num_transactions = ds->count;

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

    // EUCS
    eucs = calloc(promising_count, sizeof(double*));
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *p_items = malloc(sizeof(uint32_t) * src[i].count);
        if (!p_items) continue;
        size_t p_cnt = 0;
        for (size_t j = 0; j < src[i].count; j++) if (rank[src[i].items[j].id] != 0xFFFFFFFF) p_items[p_cnt++] = src[i].items[j].id;
        for (size_t j = 0; j < p_cnt; j++) {
            uint32_t rj = rank[p_items[j]];
            if (!eucs[rj]) eucs[rj] = calloc(promising_count, sizeof(double));
            for (size_t k = j + 1; k < p_cnt; k++) {
                uint32_t rk = rank[p_items[k]];
                if (!eucs[rk]) eucs[rk] = calloc(promising_count, sizeof(double));
                eucs[rj][rk] += src[i].total_utility;
                eucs[rk][rj] += src[i].total_utility;
            }
        }
        free(p_items);
    }

    initial_lists = malloc(sizeof(GHUI_UtilityList*) * promising_count);
    item_tidsets = malloc(sizeof(GHUI_BitSet*) * promising_count);
    for (size_t i = 0; i < promising_count; i++) {
        initial_lists[i] = calloc(1, sizeof(GHUI_UtilityList));
        initial_lists[i]->item = (uint32_t)i;
        initial_lists[i]->tuples = malloc(sizeof(GHUI_Tuple) * 16);
        item_tidsets[i] = bs_create(num_transactions);
    }

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = malloc(sizeof(uint32_t) * src[i].count);
        double *t_utils = malloc(sizeof(double) * src[i].count);
        if (!t_items || !t_utils) {
            free(t_items);
            free(t_utils);
            continue;
        }
        size_t t_count = 0;
        for (size_t j = 0; j < src[i].count; j++) {
            if (rank[src[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = src[i].items[j].id;
                t_utils[t_count] = src[i].items[j].utility;
                t_count++;
            }
        }
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
            GHUI_UtilityList *el = initial_lists[r];
            if (el->count % 16 == 0 && el->count > 0) el->tuples = realloc(el->tuples, sizeof(GHUI_Tuple) * (el->count + 16));
            el->tuples[el->count].tid = (uint32_t)i;
            el->tuples[el->count].iutil = t_utils[j];
            el->tuples[el->count].rutil = remaining;
            el->sum_iutil += t_utils[j];
            el->sum_rutil += remaining;
            el->count++;
            bs_set(item_tidsets[r], i);
            remaining += t_utils[j];
        }
        free(t_items);
        free(t_utils);
    }

    // Initialize crit for single items
    for (size_t i = 0; i < promising_count; i++) {
        initial_lists[i]->crit = malloc(sizeof(GHUI_BitSet*));
        initial_lists[i]->crit[0] = bs_create(num_transactions);
        // crit({i}, i) = g(empty) \ g(i). g(empty) is all transactions.
        for (size_t b = 0; b < initial_lists[i]->crit[0]->size; b++) {
            initial_lists[i]->crit[0]->bits[b] = ~item_tidsets[i]->bits[b];
        }
    }

    hug_count = 0;
    lug_count = 0;
    for (size_t i = 0; i < promising_count; i++) if (initial_lists[i]->sum_iutil >= min_util) hug_count++;
    
    search(NULL, initial_lists, promising_count, -1, min_util);

    printf("[HUG-Miner] Found %zu High Utility Generators.\n", hug_count);

    // Cleanup
    for (size_t i = 0; i < promising_count; i++) {
        free_ul(initial_lists[i], -1);
        bs_free(item_tidsets[i]);
        free(eucs[i]);
    }
    free(initial_lists); free(item_tidsets); free(eucs); free(promising_items); free(rank); free(twu);

    dm_bench_record_results(hug_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm hug_miner_algo = {
    .id = "hug_miner",
    .name = "HUG-Miner",
    .description = "Mining High Utility Generators using Utility-Lists and Bitsets.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(hug_miner_algo)
