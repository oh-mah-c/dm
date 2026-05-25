#include "algorithms/sfui_uf.h"
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
} SFUI_Tuple;

typedef struct {
    uint32_t item;
    SFUI_Tuple *tuples;
    size_t count;
    double sum_iutil;
    double sum_rutil;
} SFUI_UtilityList;

typedef struct {
    uint32_t *items;
    size_t length;
    double utility;
    size_t frequency;
} SFUI_Candidate;

/* --- CONTEXT --- */

static double *mua = NULL; // Max Utility Array [1..f_max]
static size_t f_max = 0;
static double mus = 0;

static SFUI_Candidate *psfuis = NULL;
static size_t psfui_count = 0;
static size_t psfui_cap = 0;

/* --- UTILS --- */

static void add_psfui(uint32_t *items, size_t len, double utility, size_t freq) {
    // 1. Check if new one is dominated by any existing
    for (size_t i = 0; i < psfui_count; i++) {
        if (psfuis[i].items == NULL) continue;
        if ((psfuis[i].frequency >= freq && psfuis[i].utility > utility) ||
            (psfuis[i].frequency > freq && psfuis[i].utility >= utility)) {
            return; // Dominated
        }
    }
    
    // 2. Remove any existing dominated by new one
    for (size_t i = 0; i < psfui_count; i++) {
        if (psfuis[i].items == NULL) continue;
        if ((freq >= psfuis[i].frequency && utility > psfuis[i].utility) ||
            (freq > psfuis[i].frequency && utility >= psfuis[i].utility)) {
            free(psfuis[i].items);
            psfuis[i].items = NULL;
        }
    }

    // 3. Update MUA
    if (utility > mua[freq]) {
        for (size_t f = 1; f <= freq; f++) {
            if (utility > mua[f]) mua[f] = utility;
        }
    }
    
    // 4. Add new one
    if (psfui_count >= psfui_cap) {
        psfui_cap = psfui_cap == 0 ? 16 : psfui_cap * 2;
        psfuis = realloc(psfuis, sizeof(SFUI_Candidate) * psfui_cap);
    }
    psfuis[psfui_count].items = malloc(sizeof(uint32_t) * len);
    memcpy(psfuis[psfui_count].items, items, sizeof(uint32_t) * len);
    psfuis[psfui_count].length = len;
    psfuis[psfui_count].utility = utility;
    psfuis[psfui_count].frequency = freq;
    psfui_count++;
}

static SFUI_UtilityList* construct(SFUI_UtilityList *P, SFUI_UtilityList *Px, SFUI_UtilityList *Py) {
    SFUI_UtilityList *Pxy = calloc(1, sizeof(SFUI_UtilityList));
    size_t cap = Px->count < Py->count ? Px->count : Py->count;
    Pxy->tuples = malloc(sizeof(SFUI_Tuple) * cap);
    
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

static void free_ul(SFUI_UtilityList *ul) {
    if (!ul) return;
    free(ul->tuples);
    free(ul);
}

/* --- LOGIC --- */

static void p_miner(SFUI_UtilityList *P, SFUI_UtilityList **extensions, size_t ext_count, uint32_t *prefix, size_t prefix_len) {
    for (size_t i = 0; i < ext_count; i++) {
        SFUI_UtilityList *Px = extensions[i];
        
        // MUE Pruning: Theorem 4 (u(extension) >= u(parent))
        // Since we are searching SFUIs, u(Y) < u(X) means Y is dominated by X.
        if (P && Px->sum_iutil < P->sum_iutil) continue;
        
        uint32_t *current_prefix = (uint32_t *)malloc(sizeof(uint32_t) * (prefix_len + 1));
        if (!current_prefix) continue;
        memcpy(current_prefix, prefix, sizeof(uint32_t) * prefix_len);
        current_prefix[prefix_len] = Px->item;

        if (Px->sum_iutil >= mua[Px->count]) {
            add_psfui(current_prefix, prefix_len + 1, Px->sum_iutil, Px->count);
        }
        
        // MUA Pruning: Theorem 2
        if (Px->sum_iutil + Px->sum_rutil < mua[Px->count]) continue;
        
        SFUI_UtilityList **next_extensions = malloc(sizeof(SFUI_UtilityList*) * (ext_count - i - 1));
        size_t next_ext_count = 0;
        
        for (size_t j = i + 1; j < ext_count; j++) {
            SFUI_UtilityList *Pxy = construct(P, Px, extensions[j]);
            Pxy->item = extensions[j]->item;
            if (Pxy->count > 0) {
                next_extensions[next_ext_count++] = Pxy;
            } else {
                free_ul(Pxy);
            }
        }
        
        if (next_ext_count > 0) {
            p_miner(Px, next_extensions, next_ext_count, current_prefix, prefix_len + 1);
        }
        
        for (size_t k = 0; k < next_ext_count; k++) free_ul(next_extensions[k]);
        free(next_extensions);
        free(current_prefix);
    }
}

static int cmp_items_twu(void *arg, const void *a, const void *b) {
    uint32_t i1 = *(uint32_t *)a;
    uint32_t i2 = *(uint32_t *)b;
    double *twu = (double *)arg;
    if (twu[i1] < twu[i2]) return -1;
    if (twu[i1] > twu[i2]) return 1;
    return (int)i1 - (int)i2;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    printf("[SFUI-UF] Mining Skyline Frequent-Utility Itemsets with Utility Filtering...\n");
    
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    double *item_utils = calloc(ds->max_id + 1, sizeof(double));
    size_t *item_freqs = calloc(ds->max_id + 1, sizeof(size_t));
    
    f_max = 0;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            uint32_t it = src[i].items[j].id;
            twu[it] += src[i].total_utility;
            item_utils[it] += src[i].items[j].utility;
            item_freqs[it]++;
            if (item_freqs[it] > f_max) f_max = item_freqs[it];
        }
    }
    
    // MUS: Theorem 3
    mus = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_freqs[i] == f_max) {
            if (item_utils[i] > mus) mus = item_utils[i];
        }
    }
    
    // MUA initialization
    mua = malloc(sizeof(double) * (f_max + 1));
    for (size_t f = 0; f <= f_max; f++) mua[f] = mus;
    
    uint32_t *promising_items = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    size_t promising_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] >= mus) {
            promising_items[promising_count++] = i;
        }
    }
    qsort_s(promising_items, promising_count, sizeof(uint32_t), cmp_items_twu, twu);
    
    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < promising_count; i++) rank[promising_items[i]] = (uint32_t)i;
    
    SFUI_UtilityList **initial_lists = malloc(sizeof(SFUI_UtilityList*) * promising_count);
    for (size_t i = 0; i < promising_count; i++) {
        initial_lists[i] = calloc(1, sizeof(SFUI_UtilityList));
        initial_lists[i]->item = promising_items[i];
        initial_lists[i]->tuples = malloc(sizeof(SFUI_Tuple) * 16);
    }
    
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = (uint32_t *)malloc(sizeof(uint32_t) * src[i].count);
        double *t_utils = (double *)malloc(sizeof(double) * src[i].count);
        size_t t_cnt = 0;
        if (!t_items || !t_utils) {
            free(t_items);
            free(t_utils);
            continue;
        }
        for (size_t j = 0; j < src[i].count; j++) {
            if (rank[src[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_cnt] = src[i].items[j].id;
                t_utils[t_cnt] = src[i].items[j].utility;
                t_cnt++;
            }
        }
        for (size_t j = 0; j < t_cnt; j++) {
            for (size_t k = j + 1; k < t_cnt; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t ti = t_items[j]; t_items[j] = t_items[k]; t_items[k] = ti;
                    double tu = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tu;
                }
            }
        }
        double remaining = 0;
        for (size_t j = t_cnt; j-- > 0; ) {
            uint32_t r = rank[t_items[j]];
            SFUI_UtilityList *el = initial_lists[r];
            if (el->count % 16 == 0 && el->count > 0) el->tuples = realloc(el->tuples, sizeof(SFUI_Tuple) * (el->count + 16));
            el->tuples[el->count].tid = (uint32_t)i;
            el->tuples[el->count].iutil = t_utils[j];
            el->tuples[el->count].rutil = remaining;
            el->sum_iutil += t_utils[j];
            el->sum_rutil += remaining;
            el->count++;
            remaining += t_utils[j];
        }
        free(t_items);
        free(t_utils);
    }
    
    psfui_count = 0;
    p_miner(NULL, initial_lists, promising_count, NULL, 0);
    
    // Final S-Miner: Filter actual SFUIs from PSFUIs
    size_t final_count = 0;
    for (size_t i = 0; i < psfui_count; i++) {
        if (psfuis[i].items == NULL) continue;
        final_count++;
    }

    printf("[SFUI-UF] Found %zu Skyline Itemsets.\n", final_count);

    // Cleanup
    for (size_t i = 0; i < psfui_count; i++) free(psfuis[i].items);
    free(psfuis); 
    for (size_t i = 0; i < promising_count; i++) free_ul(initial_lists[i]);
    free(initial_lists); free(mua); free(twu); free(item_utils); free(item_freqs); free(promising_items); free(rank);
    
    dm_bench_record_results(final_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm sfui_uf_algo = {
    .id = "sfui_uf",
    .name = "SFUI-UF",
    .description = "Skyline Frequent-Utility Itemsets with Utility Filtering (MUA, MUS, MUE).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(sfui_uf_algo)
