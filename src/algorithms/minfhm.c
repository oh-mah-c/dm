#include "algorithms/minfhm.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} MinFHM_Tuple;

typedef struct {
    uint32_t item;
    MinFHM_Tuple *tuples;
    size_t count;
    double sum_iutil;
    double sum_rutil;
} MinFHM_UtilityList;

typedef struct {
    uint32_t *items;
    size_t length;
} MinHUI_Entry;

/* --- CONTEXT --- */

static size_t minhui_count = 0;
static double **eucs = NULL;
static uint32_t promising_count = 0;
static uint32_t *promising_items = NULL;
static MinFHM_UtilityList **initial_lists = NULL;

// MinHUI-Store: simple array of entries
static MinHUI_Entry *minhui_store = NULL;
static size_t store_size = 0;
static size_t store_cap = 0;

/* --- UTILS --- */

static bool is_subset(uint32_t *small, size_t small_len, uint32_t *large, size_t large_len) {
    if (small_len > large_len) return false;
    size_t i = 0, j = 0;
    while (i < small_len && j < large_len) {
        if (small[i] == large[j]) { i++; j++; }
        else if (small[i] < large[j]) return false;
        else j++;
    }
    return i == small_len;
}

static void add_to_store(uint32_t *items, size_t len) {
    if (store_size >= store_cap) {
        store_cap = store_cap == 0 ? 16 : store_cap * 2;
        minhui_store = realloc(minhui_store, sizeof(MinHUI_Entry) * store_cap);
    }
    minhui_store[store_size].items = malloc(sizeof(uint32_t) * len);
    memcpy(minhui_store[store_size].items, items, sizeof(uint32_t) * len);
    minhui_store[store_size].length = len;
    store_size++;
    minhui_count++;
}

/* --- LOGIC --- */

static MinFHM_UtilityList* construct(MinFHM_UtilityList *P, MinFHM_UtilityList *Px, MinFHM_UtilityList *Py) {
    MinFHM_UtilityList *Pxy = calloc(1, sizeof(MinFHM_UtilityList));
    size_t cap = Px->count < Py->count ? Px->count : Py->count;
    Pxy->tuples = malloc(sizeof(MinFHM_Tuple) * cap);
    
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
            Pxy->sum_iutil += iutil;
            Pxy->sum_rutil += Py->tuples[iy].rutil;
            Pxy->count++;
            ix++; iy++;
        } else if (Px->tuples[ix].tid < Py->tuples[iy].tid) ix++;
        else iy++;
    }
    return Pxy;
}

static void free_ul(MinFHM_UtilityList *ul) {
    if (!ul) return;
    free(ul->tuples);
    free(ul);
}

static void minfhm_search(MinFHM_UtilityList *P, MinFHM_UtilityList **extensions, size_t ext_count, uint32_t *prefix, size_t prefix_len, double min_util) {
    for (size_t i = 0; i < ext_count; i++) {
        MinFHM_UtilityList *Px = extensions[i];
        uint32_t current_prefix[prefix_len + 1];
        memcpy(current_prefix, prefix, sizeof(uint32_t) * prefix_len);
        current_prefix[prefix_len] = Px->item;

        if (Px->sum_iutil >= min_util) {
            // Found a HUI. Is it minimal?
            // Since we follow depth-first and Property 5 (no superset of MinHUI is MinHUI),
            // if we reached here, it means no subset of prefix was a MinHUI.
            // But we must check subsets formed by skipping items.
            bool minimal = true;
            for (size_t s = 0; s < store_size; s++) {
                if (is_subset(minhui_store[s].items, minhui_store[s].length, current_prefix, prefix_len + 1)) {
                    minimal = false; break;
                }
            }
            if (minimal) {
                add_to_store(current_prefix, prefix_len + 1);
                // According to Property 5, we do NOT explore extensions of a MinHUI.
                continue; 
            }
        }
        
        if (Px->sum_iutil + Px->sum_rutil >= min_util) {
            MinFHM_UtilityList **next_extensions = malloc(sizeof(MinFHM_UtilityList*) * (ext_count - i - 1));
            size_t next_ext_count = 0;
            
            for (size_t j = i + 1; j < ext_count; j++) {
                MinFHM_UtilityList *Py = extensions[j];
                if (eucs[Px->item] && eucs[Px->item][Py->item] < min_util) continue;
                
                MinFHM_UtilityList *Pxy = construct(P, Px, Py);
                Pxy->item = Py->item;
                if (Pxy->sum_iutil + Pxy->sum_rutil >= min_util) {
                    next_extensions[next_ext_count++] = Pxy;
                } else {
                    free_ul(Pxy);
                }
            }
            
            if (next_ext_count > 0) {
                minfhm_search(Px, next_extensions, next_ext_count, current_prefix, prefix_len + 1, min_util);
            }
            
            for (size_t k = 0; k < next_ext_count; k++) free_ul(next_extensions[k]);
            free(next_extensions);
        }
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
    DM_MinFHM_Params *p = (DM_MinFHM_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    printf("[MinFHM] MinUtil: %.2f\n", min_util);

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    double *item_utils = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            twu[src[i].items[j].id] += src[i].total_utility;
            item_utils[src[i].items[j].id] += src[i].items[j].utility;
        }
    }

    promising_items = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    promising_count = 0;
    minhui_count = 0;
    store_size = 0;

    // Optimization 1: Identify single item MinHUIs
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_utils[i] >= min_util) {
            uint32_t it = i;
            add_to_store(&it, 1);
        } else if (twu[i] >= min_util) {
            promising_items[promising_count++] = i;
        }
    }
    qsort_s(promising_items, promising_count, sizeof(uint32_t), cmp_items, twu);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < promising_count; i++) rank[promising_items[i]] = (uint32_t)i;

    // EUCS
    eucs = calloc(promising_count, sizeof(double*));
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t p_items[src[i].count];
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
    }

    initial_lists = malloc(sizeof(MinFHM_UtilityList*) * promising_count);
    for (size_t i = 0; i < promising_count; i++) {
        initial_lists[i] = calloc(1, sizeof(MinFHM_UtilityList));
        initial_lists[i]->item = (uint32_t)i;
        initial_lists[i]->tuples = malloc(sizeof(MinFHM_Tuple) * 16);
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
        // Sort items by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t tmp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = tmp_i;
                    double tmp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tmp_u;
                }
            }
        }
        double remaining = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t r = rank[t_items[j]];
            MinFHM_UtilityList *el = initial_lists[r];
            if (el->count % 16 == 0 && el->count > 0) el->tuples = realloc(el->tuples, sizeof(MinFHM_Tuple) * (el->count + 16));
            el->tuples[el->count].tid = (uint32_t)i;
            el->tuples[el->count].iutil = t_utils[j];
            el->tuples[el->count].rutil = remaining;
            el->sum_iutil += t_utils[j];
            el->sum_rutil += remaining;
            el->count++;
            remaining += t_utils[j];
        }
    }

    minfhm_search(NULL, initial_lists, promising_count, NULL, 0, min_util);

    printf("[MinFHM] Found %zu Minimal High Utility Itemsets.\n", minhui_count);

    // Cleanup
    for (size_t i = 0; i < promising_count; i++) {
        free_ul(initial_lists[i]);
        free(eucs[i]);
    }
    for (size_t i = 0; i < store_size; i++) free(minhui_store[i].items);
    free(minhui_store); free(initial_lists); free(eucs); free(promising_items); free(rank); free(twu); free(item_utils);

    dm_bench_record_results(minhui_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm minfhm_algo = {
    .id = "minfhm",
    .name = "MinFHM",
    .description = "Mining Minimal High-Utility Itemsets with Property 5 pruning.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(minfhm_algo)
