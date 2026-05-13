#include "algorithms/tku_ce_plus.h"
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

static int cmp_doubles_desc(const void *a, const void *b) {
    double diff = *(double*)b - *(double*)a;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    printf("Starting TKU-CE+ run...\n"); fflush(stdout);
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_TKU_CE_Plus_Params *p = (DM_TKU_CE_Plus_Params *)params;
    
    int k = p ? p->k : 10;
    int N = p ? p->n : 2000;
    double rho = p ? p->rho : 0.2;
    int max_iter = p ? p->max_iter : 2000;

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    
    // 1. Initial Statistics & CUV Pruning
    double *item_utils = calloc(ds->max_id + 1, sizeof(double));
    double *item_twus = calloc(ds->max_id + 1, sizeof(double));
    bool *item_exists = calloc(ds->max_id + 1, sizeof(bool));
    size_t distinct_items = 0;

    for (size_t i = 0; i < ds->count; i++) {
        double tu = 0;
        for (size_t j = 0; j < src[i].count; j++) tu += src[i].items[j].utility;
        for (size_t j = 0; j < src[i].count; j++) {
            uint32_t id = src[i].items[j].id;
            item_utils[id] += src[i].items[j].utility;
            item_twus[id] += tu;
            if (!item_exists[id]) {
                item_exists[id] = true;
                distinct_items++;
            }
        }
    }

    double *utils_sorted = malloc(sizeof(double) * distinct_items);
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_exists[i]) utils_sorted[idx++] = item_utils[i];
    }
    qsort(utils_sorted, distinct_items, sizeof(double), cmp_doubles_desc);

    double CUV = (k <= (int)distinct_items) ? utils_sorted[k - 1] : 0;
    free(utils_sorted);

    // Filter items based on TWU < CUV
    uint32_t *filtered_ids = malloc(sizeof(uint32_t) * distinct_items);
    size_t m = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_exists[i]) {
            if (item_twus[i] >= CUV) {
                filtered_ids[m++] = i;
            }
        }
    }
    printf("CUV = %.2f. Items pruned: %zu -> %zu\n", CUV, distinct_items, m);
    free(item_utils); free(item_twus); free(item_exists);

    ItemInfo *items = malloc(sizeof(ItemInfo) * m);
    for (size_t i = 0; i < m; i++) {
        items[i].id = filtered_ids[i];
        items[i].bitset = dm_bitset_create(ds->count);
        items[i].utilities = calloc(ds->count, sizeof(double));
        for (size_t t = 0; t < ds->count; t++) {
            for (size_t j = 0; j < src[t].count; j++) {
                if (src[t].items[j].id == filtered_ids[i]) {
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

    int rho_N = (int)ceil(rho * N);
    if (rho_N < 1) rho_N = 1;

    for (int iter = 0; iter < max_iter; iter++) {
        // Smoothing Mutation (Eq 7)
        double alpha = 0;
        if (iter > 0 && samples[0].utility > 0) {
            alpha = ((samples[0].utility - samples[rho_N - 1].utility) / samples[0].utility) * rho;
        }
        int num_random = (int)floor(N * alpha);
        
        // Sample Refinement: Retain elite samples from previous iteration
        // The first rho_N samples in the sorted 'samples' array are elite.
        // We only generate N - rho_N new samples.
        
        if (iter > 0) {
            // Move elite samples to a temp storage if needed, or just offset the generation
            // Actually, we can just keep the first rho_N samples and generate N - rho_N new ones.
            // But we need to make sure we don't overwrite them.
            
            // Generate N - rho_N new samples starting from index rho_N
            for (int i = rho_N; i < N; i++) {
                if (i < rho_N + num_random) {
                    for (size_t j = 0; j < m; j++) samples[i].bits[j] = (rand() % 2);
                } else {
                    for (size_t j = 0; j < m; j++) {
                        double r = (double)rand() / RAND_MAX;
                        samples[i].bits[j] = (r <= prob_vec[j]) ? 1 : 0;
                    }
                }
                samples[i].utility = calculate_utility(samples[i].bits, m, items, ds->count, temp_bitset);
                topk_list_update(&topk, samples[i].bits, m, filtered_ids, samples[i].utility);
            }
        } else {
            // First iteration: generate all N samples
            for (int i = 0; i < N; i++) {
                for (size_t j = 0; j < m; j++) {
                    double r = (double)rand() / RAND_MAX;
                    samples[i].bits[j] = (r <= prob_vec[j]) ? 1 : 0;
                }
                samples[i].utility = calculate_utility(samples[i].bits, m, items, ds->count, temp_bitset);
                topk_list_update(&topk, samples[i].bits, m, filtered_ids, samples[i].utility);
            }
        }

        qsort(samples, N, sizeof(Sample), cmp_samples);

        double gamma = samples[rho_N - 1].utility;

        if (iter % 10 == 0) {
            printf("Iteration %d: alpha = %.4f, gamma = %.2f, Max = %.2f\n", iter, alpha, gamma, samples[0].utility);
            fflush(stdout);
        }

        bool changed = false;
        for (size_t j = 0; j < m; j++) {
            double sum_indicator = 0;
            double sum_indicator_success = 0;
            for (int i = 0; i < N; i++) {
                if (samples[i].utility >= gamma && samples[i].utility > 0) {
                    sum_indicator += 1.0;
                    if (samples[i].bits[j]) sum_indicator_success += 1.0;
                }
            }
            double next_p = (sum_indicator > 0) ? (sum_indicator_success / sum_indicator) : prob_vec[j];
            // Small smoothing as in typical CE (not explicitly in paper but helps)
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
    free(filtered_ids);
    for (size_t i = 0; i < topk.count; i++) free(topk.results[i].items);
    free(topk.results);

    return DM_SUCCESS;
}

DM_Algorithm tku_ce_plus_algo = {
    .id = "tku_ce_plus",
    .name = "TKU-CE+",
    .description = "Improved Cross-Entropy Method for Mining Top-K High Utility Itemsets (Song et al. 2021).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(tku_ce_plus_algo)
