#include "algorithms/sfu_ce.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include "core/dm_bitset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t *items;
    size_t length;
    double utility;
    size_t frequency;
} SFU_CE_Candidate;

/* --- CONTEXT --- */

static SFU_CE_Candidate *csfui = NULL;
static size_t csfui_count = 0;
static size_t csfui_cap = 0;

/* --- UTILS --- */

static double get_utility_fast(uint32_t *iv, size_t n, DM_Trans_Utility *src, size_t ds_count, size_t *out_freq, DM_BitSet **item_bits, DM_BitSet *match_bits) {
    dm_bitset_set_all(match_bits);
    
    bool empty = true;
    for (size_t j = 0; j < n; j++) {
        if (iv[j]) {
            dm_bitset_and(match_bits, item_bits[j]);
            empty = false;
        }
    }
    
    if (empty) return 0;
    
    double total_util = 0;
    size_t freq = 0;
    for (size_t i = 0; i < ds_count; i++) {
        if (dm_bitset_get(match_bits, i)) {
            freq++;
            for (size_t j = 0; j < src[i].count; j++) {
                if (iv[src[i].items[j].id]) total_util += src[i].items[j].utility;
            }
        }
    }
    
    if (out_freq) *out_freq = freq;
    return total_util;
}

static void filter_sfui(uint32_t *iv, size_t n, double utility, size_t freq) {
    if (utility == 0) return;
    
    for (size_t i = 0; i < csfui_count; i++) {
        if (csfui[i].items == NULL) continue;
        if ((csfui[i].frequency >= freq && csfui[i].utility > utility) ||
            (csfui[i].frequency > freq && csfui[i].utility >= utility)) return;
    }
    
    for (size_t i = 0; i < csfui_count; i++) {
        if (csfui[i].items == NULL) continue;
        if ((freq >= csfui[i].frequency && utility > csfui[i].utility) ||
            (freq > csfui[i].frequency && utility >= csfui[i].utility)) {
            free(csfui[i].items);
            csfui[i].items = NULL;
        }
    }
    
    if (csfui_count >= csfui_cap) {
        csfui_cap = csfui_cap == 0 ? 16 : csfui_cap * 2;
        csfui = realloc(csfui, sizeof(SFU_CE_Candidate) * csfui_cap);
    }
    size_t len = 0;
    for (size_t i = 0; i < n; i++) if (iv[i]) len++;
    csfui[csfui_count].items = malloc(sizeof(uint32_t) * len);
    size_t idx = 0;
    for (size_t i = 0; i < n; i++) if (iv[i]) csfui[csfui_count].items[idx++] = (uint32_t)i;
    csfui[csfui_count].length = len;
    csfui[csfui_count].utility = utility;
    csfui[csfui_count].frequency = freq;
    csfui_count++;
}

static int cmp_samples(const void *a, const void *b) {
    double u1 = *(double*)a;
    double u2 = *(double*)b;
    if (u1 > u2) return -1;
    if (u1 < u2) return 1;
    return 0;
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_SFU_CE_Params *p = (DM_SFU_CE_Params *)params;
    int sample_size = p ? p->sample_size : 1000;
    int max_iters = p ? p->max_iterations : 100;
    double rho = p ? p->quantile : 0.1;
    double alpha = p ? p->mutation_factor : 0.2;
    
    printf("[SFU-CE] Mining Skyline HUIs using Cross-Entropy (Heuristic)...\n");
    srand((unsigned int)time(NULL));

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    size_t n_items = ds->max_id + 1;
    
    DM_BitSet **item_bits = malloc(sizeof(DM_BitSet*) * n_items);
    for (size_t i = 0; i < n_items; i++) item_bits[i] = dm_bitset_create(ds->count);
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) dm_bitset_set(item_bits[src[i].items[j].id], i);
    }
    
    DM_BitSet *match_bits = dm_bitset_create(ds->count);

    double *pv = malloc(sizeof(double) * n_items);
    for (size_t i = 0; i < n_items; i++) pv[i] = 2.0 / (double)n_items;
    
    uint32_t **samples = malloc(sizeof(uint32_t*) * sample_size);
    double *sample_utils = malloc(sizeof(double) * sample_size);
    size_t *sample_freqs = malloc(sizeof(size_t) * sample_size);
    for (int i = 0; i < sample_size; i++) samples[i] = malloc(sizeof(uint32_t) * n_items);

    csfui_count = 0;

    for (int iter = 0; iter < max_iters; iter++) {
        int mutation_count = (int)(sample_size * alpha);
        int valid_samples = 0;
        for (int i = 0; i < sample_size; i++) {
            memset(samples[i], 0, sizeof(uint32_t) * n_items);
            if (i < mutation_count && iter > 0) {
                uint32_t pool[n_items]; size_t pool_size = 0;
                for (size_t j = 0; j < n_items; j++) if (pv[j] > 0.5) pool[pool_size++] = (uint32_t)j;
                if (pool_size > 0) {
                    int num = (rand() % (int)pool_size) + 1;
                    for (int k = 0; k < num; k++) samples[i][pool[rand() % pool_size]] = 1;
                }
            } else {
                for (size_t j = 0; j < n_items; j++) {
                    if ((double)rand() / RAND_MAX < pv[j]) samples[i][j] = 1;
                }
            }
            sample_utils[i] = get_utility_fast(samples[i], n_items, src, ds->count, &sample_freqs[i], item_bits, match_bits);
            if (sample_utils[i] > 0) {
                filter_sfui(samples[i], n_items, sample_utils[i], sample_freqs[i]);
                valid_samples++;
            }
        }
        
        double sorted_utils[sample_size];
        memcpy(sorted_utils, sample_utils, sizeof(double) * sample_size);
        qsort(sorted_utils, sample_size, sizeof(double), cmp_samples);
        double gamma = sorted_utils[(int)(rho * (sample_size - 1))];
        
        double denom = 0;
        for (int i = 0; i < sample_size; i++) if (sample_utils[i] >= gamma && sample_utils[i] > 0) denom += 1.0;
        
        if (denom > 0) {
            for (size_t j = 0; j < n_items; j++) {
                double num = 0;
                for (int i = 0; i < sample_size; i++) if (sample_utils[i] >= gamma && sample_utils[i] > 0 && samples[i][j]) num += 1.0;
                pv[j] = (num / denom) * 0.7 + pv[j] * 0.3;
            }
        }
        
        if (iter % 10 == 0) printf("  Iteration %d: %d valid samples, gamma=%.2f, psfui=%zu\n", iter, valid_samples, gamma, csfui_count);
    }

    size_t final_count = 0;
    for (size_t i = 0; i < csfui_count; i++) if (csfui[i].items) final_count++;

    printf("[SFU-CE] Found %zu Potential Skyline Itemsets (Heuristic).\n", final_count);

    for (int i = 0; i < sample_size; i++) free(samples[i]);
    free(samples); free(sample_utils); free(sample_freqs); free(pv);
    for (size_t i = 0; i < n_items; i++) dm_bitset_free(item_bits[i]);
    free(item_bits); dm_bitset_free(match_bits);
    for (size_t i = 0; i < csfui_count; i++) if (csfui[i].items) free(csfui[i].items);
    free(csfui);

    dm_bench_record_results(final_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm sfu_ce_algo = {
    .id = "sfu_ce",
    .name = "SFU-CE",
    .description = "Skyline Frequent-Utility Discovery using Cross-Entropy (Heuristic).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(sfu_ce_algo)
