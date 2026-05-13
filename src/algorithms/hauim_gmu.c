#include "algorithms/hauim_gmu.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t *tids;
    size_t count;
} TIDSet;

typedef struct {
    double *gmu_sums; // gmu_sums[k-1] = sum of top k utilities in this transaction
    size_t item_count;
} TransGMU;

/* --- UTILS --- */

static int cmp_double_desc(const void *a, const void *b) {
    double v1 = *(double*)a;
    double v2 = *(double*)b;
    if (v1 > v2) return -1;
    if (v1 < v2) return 1;
    return 0;
}

static TIDSet intersect(TIDSet s1, TIDSet s2) {
    TIDSet res;
    res.tids = malloc(sizeof(uint32_t) * (s1.count < s2.count ? s1.count : s2.count));
    res.count = 0;
    size_t i = 0, j = 0;
    while (i < s1.count && j < s2.count) {
        if (s1.tids[i] == s2.tids[j]) {
            res.tids[res.count++] = s1.tids[i];
            i++; j++;
        } else if (s1.tids[i] < s2.tids[j]) i++;
        else j++;
    }
    return res;
}

/* --- SEARCH --- */

static size_t haui_count = 0;
static size_t total_items = 0;
static size_t csup = 0;
static double threshold = 0;
static TransGMU *trans_gmu = NULL;
static DM_Trans_Utility *data = NULL;
static uint32_t *promising_items = NULL;
static size_t promising_count = 0;
static TIDSet *item_tidsets = NULL;

static void search(uint32_t *prefix, size_t prefix_len, TIDSet prefix_tids, size_t last_idx) {
    for (size_t i = last_idx + 1; i < promising_count; i++) {
        TIDSet new_tids = intersect(prefix_tids, item_tidsets[i]);
        
        if (new_tids.count >= csup) {
            size_t new_len = prefix_len + 1;
            double sum_gmu = 0;
            for (size_t j = 0; j < new_tids.count; j++) {
                uint32_t tid = new_tids.tids[j];
                size_t k = (new_len > trans_gmu[tid].item_count) ? trans_gmu[tid].item_count : new_len;
                sum_gmu += trans_gmu[tid].gmu_sums[k - 1];
            }
            
            if (sum_gmu / new_len >= threshold) {
                // Potential HAUI, verify
                double total_u = 0;
                uint32_t *itemset = malloc(sizeof(uint32_t) * new_len);
                memcpy(itemset, prefix, sizeof(uint32_t) * prefix_len);
                itemset[prefix_len] = promising_items[i];
                
                for (size_t j = 0; j < new_tids.count; j++) {
                    uint32_t tid = new_tids.tids[j];
                    for (size_t l = 0; l < new_len; l++) {
                        for (size_t m = 0; l < new_len && m < data[tid].count; m++) {
                            if (data[tid].items[m].id == itemset[l]) {
                                total_u += data[tid].items[m].utility;
                                break;
                            }
                        }
                    }
                }
                
                if (total_u / new_len >= threshold) {
                    haui_count++;
                    total_items += new_len;
                }
                
                // Recurse
                search(itemset, new_len, new_tids, i);
                free(itemset);
            }
        }
        free(new_tids.tids);
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_HAUIM_GMU_Params *p = (DM_HAUIM_GMU_Params *)params;
    double ratio = p ? p->min_utility_ratio : 0.01;
    
    data = (DM_Trans_Utility *)ds->payload;
    
    // 1. Calculate TU and pre-calculate GMU sums for each transaction
    double total_utility = 0;
    double *all_mu = malloc(sizeof(double) * ds->count);
    trans_gmu = malloc(sizeof(TransGMU) * ds->count);
    
    for (size_t i = 0; i < ds->count; i++) {
        total_utility += data[i].total_utility;
        trans_gmu[i].item_count = data[i].count;
        trans_gmu[i].gmu_sums = malloc(sizeof(double) * data[i].count);
        
        double *utils = malloc(sizeof(double) * data[i].count);
        for (size_t j = 0; j < data[i].count; j++) utils[j] = data[i].items[j].utility;
        qsort(utils, data[i].count, sizeof(double), cmp_double_desc);
        
        double sum = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            sum += utils[j];
            trans_gmu[i].gmu_sums[j] = sum;
        }
        all_mu[i] = (data[i].count > 0) ? utils[0] : 0;
        free(utils);
    }

    threshold = ratio * total_utility;
    printf("[HAUIM-GMU] Ratio: %.4f, Total Utility: %.2f, Threshold: %.2f\n", ratio, total_utility, threshold);

    // 2. Calculate csup
    qsort(all_mu, ds->count, sizeof(double), cmp_double_desc);
    double cumulative_mu = 0;
    csup = 0;
    for (size_t i = 0; i < ds->count; i++) {
        cumulative_mu += all_mu[i];
        if (cumulative_mu >= threshold) {
            csup = i + 1;
            break;
        }
    }
    printf("[HAUIM-GMU] Critical Support Count (csup): %zu\n", csup);

    // 3. Find 1-HGAUUBIs and build item tidsets
    item_tidsets = calloc(ds->max_id + 1, sizeof(TIDSet));
    size_t *item_sups = calloc(ds->max_id + 1, sizeof(size_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j].id;
            item_sups[id]++;
        }
    }
    
    promising_items = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    promising_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_sups[i] >= csup) {
            // Check GAUUB for 1-itemset: sum mu(T) where i in T >= threshold
            double sum_mu = 0;
            for (size_t j = 0; j < ds->count; j++) {
                bool found = false;
                for (size_t k = 0; k < data[j].count; k++) {
                    if (data[j].items[k].id == i) { found = true; break; }
                }
                if (found) {
                    sum_mu += trans_gmu[j].gmu_sums[0];
                }
            }
            
            if (sum_mu >= threshold) {
                promising_items[promising_count++] = i;
                item_tidsets[promising_count - 1].tids = malloc(sizeof(uint32_t) * item_sups[i]);
                item_tidsets[promising_count - 1].count = 0;
            }
        }
    }
    
    // Fill tidsets
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j].id;
            // Check if promising
            for (size_t k = 0; k < promising_count; k++) {
                if (promising_items[k] == id) {
                    item_tidsets[k].tids[item_tidsets[k].count++] = (uint32_t)i;
                    break;
                }
            }
        }
    }

    // 4. Search
    haui_count = 0;
    total_items = 0;
    TIDSet root_tids;
    root_tids.count = ds->count;
    root_tids.tids = malloc(sizeof(uint32_t) * ds->count);
    for (size_t i = 0; i < ds->count; i++) root_tids.tids[i] = (uint32_t)i;
    
    search(NULL, 0, root_tids, -1);

    printf("[HAUIM-GMU] Found %zu HAUIs\n", haui_count);
    dm_bench_record_results(haui_count, total_items);

    // Cleanup
    free(root_tids.tids);
    for (size_t i = 0; i < promising_count; i++) free(item_tidsets[i].tids);
    free(item_tidsets);
    for (size_t i = 0; i < ds->count; i++) free(trans_gmu[i].gmu_sums);
    free(trans_gmu);
    free(promising_items); free(item_sups); free(all_mu);

    return DM_SUCCESS;
}

DM_Algorithm hauim_gmu_algo = {
    .id = "hauim_gmu",
    .name = "HAUIM-GMU",
    .description = "High Average-Utility Itemset Mining based on Generalized Maximal Utility (Song et al. 2021).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(hauim_gmu_algo)
