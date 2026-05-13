#include "algorithms/tku_ce.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include "core/dm_bitset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint8_t *bits;
    double utility;
} Sample;

typedef struct {
    uint32_t id;
    DM_BitSet *bitset;
    double *utilities;
} ItemInfo;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} HUI_Result;

typedef struct {
    HUI_Result *results;
    size_t count;
    size_t k;
} TopK_List;

/* --- UTILS --- */

static void topk_list_init(TopK_List *list, size_t k) {
    list->k = k;
    list->count = 0;
    list->results = malloc(sizeof(HUI_Result) * (k + 1));
}

static void topk_list_update(TopK_List *list, uint8_t *bits, size_t m, uint32_t *item_ids, double util) {
    if (util <= 0) return;
    
    size_t len = 0;
    for (size_t i = 0; i < m; i++) if (bits[i]) len++;
    if (len == 0) return;

    for (size_t i = 0; i < list->count; i++) {
        if (list->results[i].len == len) {
            bool match = true;
            size_t k_idx = 0;
            for (size_t j = 0; j < m; j++) {
                if (bits[j]) {
                    if (list->results[i].items[k_idx++] != item_ids[j]) { match = false; break; }
                }
            }
            if (match) return;
        }
    }

    if (list->count < list->k || util > list->results[list->count - 1].utility) {
        if (list->count == list->k) {
            free(list->results[list->count - 1].items);
        } else {
            list->count++;
        }

        size_t i = list->count - 1;
        while (i > 0 && list->results[i - 1].utility < util) {
            list->results[i] = list->results[i - 1];
            i--;
        }
        
        list->results[i].items = malloc(sizeof(uint32_t) * len);
        size_t k_idx = 0;
        for (size_t j = 0; j < m; j++) if (bits[j]) list->results[i].items[k_idx++] = item_ids[j];
        list->results[i].len = len;
        list->results[i].utility = util;
    }
}

static double calculate_utility(uint8_t *bits, size_t m, ItemInfo *items, size_t n_trans, DM_BitSet *temp) {
    dm_bitset_set_all(temp);
    bool has_bits = false;
    for (size_t i = 0; i < m; i++) {
        if (bits[i]) {
            dm_bitset_and(temp, items[i].bitset);
            has_bits = true;
            if (dm_bitset_count(temp) == 0) break;
        }
    }
    if (!has_bits || dm_bitset_count(temp) == 0) return 0;

    double total_util = 0;
    for (size_t t = 0; t < n_trans; t++) {
        if (dm_bitset_get(temp, t)) {
            double tu = 0;
            for (size_t i = 0; i < m; i++) if (bits[i]) tu += items[i].utilities[t];
            total_util += tu;
        }
    }
    return total_util;
}

static int cmp_samples(const void *a, const void *b) {
    double diff = ((Sample *)b)->utility - ((Sample *)a)->utility;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    printf("Starting TKU-CE run...\n"); fflush(stdout);
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_TKU_CE_Params *p = (DM_TKU_CE_Params *)params;
    
    int k = p ? p->k : 10;
    int N = p ? p->n : 2000;
    double rho = p ? p->rho : 0.2;
    int max_iter = p ? p->max_iter : 2000;

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    
    uint32_t *item_ids = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    size_t m = 0;
    bool *seen = calloc(ds->max_id + 1, sizeof(bool));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            if (!seen[src[i].items[j].id]) {
                seen[src[i].items[j].id] = true;
                item_ids[m++] = src[i].items[j].id;
            }
        }
    }
    free(seen);

    ItemInfo *items = malloc(sizeof(ItemInfo) * m);
    for (size_t i = 0; i < m; i++) {
        items[i].id = item_ids[i];
        items[i].bitset = dm_bitset_create(ds->count);
        items[i].utilities = calloc(ds->count, sizeof(double));
        for (size_t t = 0; t < ds->count; t++) {
            for (size_t j = 0; j < src[t].count; j++) {
                if (src[t].items[j].id == item_ids[i]) {
                    dm_bitset_set(items[i].bitset, t);
                    items[i].utilities[t] = src[t].items[j].utility;
                    break;
                }
            }
        }
    }

    double *prob_vec = malloc(sizeof(double) * m);
    for (size_t i = 0; i < m; i++) prob_vec[i] = 0.5;

    Sample *samples = malloc(sizeof(Sample) * N);
    for (int i = 0; i < N; i++) samples[i].bits = malloc(m);

    TopK_List topk;
    topk_list_init(&topk, k);

    srand((unsigned int)time(NULL));
    DM_BitSet *temp_bitset = dm_bitset_create(ds->count);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < N; i++) {
            for (size_t j = 0; j < m; j++) {
                double r = (double)rand() / RAND_MAX;
                samples[i].bits[j] = (r <= prob_vec[j]) ? 1 : 0;
            }
            samples[i].utility = calculate_utility(samples[i].bits, m, items, ds->count, temp_bitset);
            topk_list_update(&topk, samples[i].bits, m, item_ids, samples[i].utility);
        }

        qsort(samples, N, sizeof(Sample), cmp_samples);

        int rho_N = (int)(rho * N);
        if (rho_N < 1) rho_N = 1;
        double gamma = samples[rho_N - 1].utility;

        if (iter % 10 == 0) {
            printf("Iteration %d: gamma = %.2f, Max Sample Utility = %.2f, Top-K Min Utility = %.2f\n", 
                   iter, gamma, samples[0].utility, topk.count > 0 ? topk.results[topk.count-1].utility : 0);
            fflush(stdout);
        }

        bool changed = false;
        for (size_t j = 0; j < m; j++) {
            double sum_indicator = 0;
            double sum_indicator_success = 0;
            for (int i = 0; i < N; i++) {
                if (samples[i].utility >= gamma) {
                    sum_indicator += 1.0;
                    if (samples[i].bits[j]) sum_indicator_success += 1.0;
                }
            }
            double next_p = (sum_indicator > 0) ? (sum_indicator_success / sum_indicator) : prob_vec[j];
            if (fabs(next_p - prob_vec[j]) > 1e-6) changed = true;
            prob_vec[j] = next_p;
        }

        if (!changed) break;

        bool is_binary = true;
        for (size_t j = 0; j < m; j++) {
            if (prob_vec[j] > 1e-6 && prob_vec[j] < (1.0 - 1e-6)) {
                is_binary = false;
                break;
            }
        }
        if (is_binary) break;
    }

    dm_bench_record_results(topk.count, 0);

    dm_bitset_free(temp_bitset);
    for (int i = 0; i < N; i++) free(samples[i].bits);
    free(samples);
    free(prob_vec);
    for (size_t i = 0; i < m; i++) {
        dm_bitset_free(items[i].bitset);
        free(items[i].utilities);
    }
    free(items);
    free(item_ids);
    for (size_t i = 0; i < topk.count; i++) free(topk.results[i].items);
    free(topk.results);

    return DM_SUCCESS;
}

DM_Algorithm tku_ce_algo = {
    .id = "tku_ce",
    .name = "TKU-CE",
    .description = "Cross-Entropy Method for Mining Top-K High Utility Itemsets (Song et al. 2021).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(tku_ce_algo)
