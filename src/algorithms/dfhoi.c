#include "algorithms/dfhoi.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    size_t length;
    uint32_t *tids;
    size_t num_tids;
} DFHOIItemset;

typedef struct {
    uint32_t orig_tid;
    uint32_t len;
} TransMeta;

static int cmp_trans_meta(const void *a, const void *b) {
    TransMeta *ta = (TransMeta *)a;
    TransMeta *tb = (TransMeta *)b;
    if (ta->len < tb->len) return -1;
    if (ta->len > tb->len) return 1;
    return 0;
}

static bool check_ubo_and_o(uint32_t *tids, size_t size, uint32_t *g_tsize, size_t k, double xi, bool *is_ho, bool hastheSameLength, uint32_t first_len) {
    if ((double)size < xi) return false;

    if (hastheSameLength) {
        double o_val = (double)(k * size) / (double)first_len;
        *is_ho = (o_val >= xi);
        return true;
    }

    double current_sum = 0.0;
    double max_ubo = 0.0;
    
    for (int i = (int)size - 1; i >= 0; i--) {
        uint32_t tsize = g_tsize[tids[i]];
        current_sum += 1.0 / (double)tsize;
        
        if (i == 0 || g_tsize[tids[i-1]] < tsize) {
            double current_ubo = tsize * current_sum;
            if (current_ubo > max_ubo) {
                max_ubo = current_ubo;
            }
        }
    }
    
    if (max_ubo >= xi) {
        *is_ho = ((k * current_sum) >= xi);
        return true;
    }
    return false;
}

static void mine_depth_hois(DFHOIItemset *classes, size_t class_size, uint32_t *g_tsize, double xi, size_t *total_ho_count, size_t *total_ho_footprint, uint32_t *buffer, bool hastheSameLength, uint32_t first_len) {
    if (class_size < 2) return;
    
    for (size_t i = 0; i < class_size; i++) {
        DFHOIItemset *P1 = &classes[i];
        
        DFHOIItemset *children = NULL;
        if (class_size - i - 1 > 0) {
            children = malloc((class_size - i - 1) * sizeof(DFHOIItemset));
        }
        size_t child_count = 0;
        
        for (size_t j = i + 1; j < class_size; j++) {
            DFHOIItemset *P2 = &classes[j];
            
            size_t p1_idx = 0, p2_idx = 0, new_idx = 0;
            
            while (p1_idx < P1->num_tids && p2_idx < P2->num_tids) {
                if (P1->tids[p1_idx] < P2->tids[p2_idx]) {
                    p1_idx++;
                } else if (P1->tids[p1_idx] > P2->tids[p2_idx]) {
                    p2_idx++;
                } else {
                    buffer[new_idx++] = P1->tids[p1_idx];
                    p1_idx++;
                    p2_idx++;
                }
            }
            
            if (new_idx >= xi) {
                bool is_ho = false;
                size_t new_k = P1->length + 1;
                if (check_ubo_and_o(buffer, new_idx, g_tsize, new_k, xi, &is_ho, hastheSameLength, first_len)) {
                    DFHOIItemset *P = &children[child_count++];
                    P->length = new_k;
                    P->num_tids = new_idx;
                    P->tids = malloc(new_idx * sizeof(uint32_t));
                    memcpy(P->tids, buffer, new_idx * sizeof(uint32_t));
                    
                    if (is_ho) {
                        (*total_ho_count)++;
                        (*total_ho_footprint) += new_k;
                    }
                }
            }
        }
        
        if (child_count > 0) {
            mine_depth_hois(children, child_count, g_tsize, xi, total_ho_count, total_ho_footprint, buffer, hastheSameLength, first_len);
            
            for (size_t c = 0; c < child_count; c++) {
                free(children[c].tids);
            }
        }
        if (children) free(children);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_DFHOI_Params *dfhoi_params = (DM_DFHOI_Params *)params;
    double min_occ_param = dfhoi_params ? dfhoi_params->min_occupancy : 0.01;
    double xi = (min_occ_param < 1.0) ? (min_occ_param * ds->count) : min_occ_param;

    printf("[DFHOI] Starting on %zu transactions. Min Occupancy Threshold: %.2f\n", ds->count, xi);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    
    TransMeta *meta = malloc(ds->count * sizeof(TransMeta));
    bool hastheSameLength = true;
    uint32_t first_len = data[0].count;
    
    for (size_t i = 0; i < ds->count; i++) {
        meta[i].orig_tid = i;
        meta[i].len = data[i].count;
        if (data[i].count != first_len) {
            hastheSameLength = false;
        }
    }
    qsort(meta, ds->count, sizeof(TransMeta), cmp_trans_meta);
    
    uint32_t *g_tsize = malloc(ds->count * sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        g_tsize[i] = meta[i].len;
    }

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < data[orig].count; j++) {
            counts[data[orig].items[j]]++;
        }
    }

    uint32_t **item_tids = malloc((ds->max_id + 1) * sizeof(uint32_t*));
    uint32_t *item_idx = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= xi) {
            item_tids[i] = malloc(counts[i] * sizeof(uint32_t));
        } else {
            item_tids[i] = NULL;
        }
    }

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < data[orig].count; j++) {
            uint32_t item = data[orig].items[j];
            if (counts[item] >= xi) {
                item_tids[item][item_idx[item]++] = i;
            }
        }
    }
    free(meta);
    free(item_idx);

    DFHOIItemset *C1 = malloc((ds->max_id + 1) * sizeof(DFHOIItemset));
    size_t c1_count = 0;
    
    size_t total_ho_count = 0;
    size_t total_ho_footprint = 0;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= xi) {
            bool is_ho = false;
            if (check_ubo_and_o(item_tids[i], counts[i], g_tsize, 1, xi, &is_ho, hastheSameLength, first_len)) {
                DFHOIItemset *it = &C1[c1_count++];
                it->length = 1;
                it->num_tids = counts[i];
                it->tids = malloc(counts[i] * sizeof(uint32_t));
                memcpy(it->tids, item_tids[i], counts[i] * sizeof(uint32_t));
                
                if (is_ho) {
                    total_ho_count++;
                    total_ho_footprint += 1;
                }
            }
        }
        if (item_tids[i]) free(item_tids[i]);
    }
    free(item_tids);
    free(counts);

    uint32_t *buffer = malloc(ds->count * sizeof(uint32_t));
    mine_depth_hois(C1, c1_count, g_tsize, xi, &total_ho_count, &total_ho_footprint, buffer, hastheSameLength, first_len);
    free(buffer);

    for (size_t i = 0; i < c1_count; i++) {
        free(C1[i].tids);
    }
    free(C1);

    free(g_tsize);
    printf("[DFHOI] Complete. Total high occupancy itemsets found: %zu\n", total_ho_count);
    dm_bench_record_results(total_ho_count, total_ho_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "dfhoi",
    .name = "DFHOI Algorithm",
    .description = "Depth First Search for High Occupancy Itemset Mining (Nguyen et al. 2023).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
