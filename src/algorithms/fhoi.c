#include "algorithms/fhoi.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint32_t parent_id;
    uint32_t item;
    uint32_t *tids;
    size_t num_tids;
} FHOIItemset;

typedef struct {
    FHOIItemset *itemsets;
    size_t count;
    size_t capacity;
} FHOIItemsetList;

static void list_init(FHOIItemsetList *list) {
    list->count = 0;
    list->capacity = 1024;
    list->itemsets = malloc(list->capacity * sizeof(FHOIItemset));
}

static void list_add(FHOIItemsetList *list, FHOIItemset itemset) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->itemsets = realloc(list->itemsets, list->capacity * sizeof(FHOIItemset));
    }
    list->itemsets[list->count++] = itemset;
}

static void list_free(FHOIItemsetList *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->itemsets[i].tids);
    }
    free(list->itemsets);
}

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

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FHOI_Params *fhoi_params = (DM_FHOI_Params *)params;
    double min_occ_param = fhoi_params ? fhoi_params->min_occupancy : 0.01;
    double xi = (min_occ_param < 1.0) ? (min_occ_param * ds->count) : min_occ_param;

    printf("[FHOI] Starting on %zu transactions. Min Occupancy Threshold: %.2f\n", ds->count, xi);

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

    FHOIItemsetList C_prev;
    list_init(&C_prev);
    
    size_t total_ho_count = 0;
    size_t total_ho_footprint = 0;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= xi) {
            bool is_ho = false;
            if (check_ubo_and_o(item_tids[i], counts[i], g_tsize, 1, xi, &is_ho, hastheSameLength, first_len)) {
                FHOIItemset it;
                it.parent_id = 0; // level 1 shared parent_id
                it.item = i;
                it.num_tids = counts[i];
                it.tids = malloc(counts[i] * sizeof(uint32_t));
                memcpy(it.tids, item_tids[i], counts[i] * sizeof(uint32_t));
                list_add(&C_prev, it);
                
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

    size_t k = 2;
    uint32_t *buffer = malloc(ds->count * sizeof(uint32_t));
    
    while (C_prev.count > 0) {
        FHOIItemsetList C_curr;
        list_init(&C_curr);
        
        for (size_t i = 0; i < C_prev.count; i++) {
            for (size_t j = i + 1; j < C_prev.count; j++) {
                FHOIItemset *P1 = &C_prev.itemsets[i];
                FHOIItemset *P2 = &C_prev.itemsets[j];
                
                if (P1->parent_id != P2->parent_id) {
                    break; // Because they are sorted by equivalence class!
                }
                
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
                    if (check_ubo_and_o(buffer, new_idx, g_tsize, k, xi, &is_ho, hastheSameLength, first_len)) {
                        FHOIItemset P;
                        P.parent_id = i;
                        P.item = P2->item;
                        P.num_tids = new_idx;
                        P.tids = malloc(new_idx * sizeof(uint32_t));
                        memcpy(P.tids, buffer, new_idx * sizeof(uint32_t));
                        list_add(&C_curr, P);
                        
                        if (is_ho) {
                            total_ho_count++;
                            total_ho_footprint += k;
                        }
                    }
                }
            }
        }
        
        list_free(&C_prev);
        C_prev = C_curr;
        k++;
    }
    
    free(buffer);
    free(g_tsize);
    printf("[FHOI] Complete. Total high occupancy itemsets found: %zu\n", total_ho_count);
    dm_bench_record_results(total_ho_count, total_ho_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fhoi",
    .name = "FHOI Algorithm",
    .description = "Fast High Occupancy Itemset Mining (Nguyen et al. 2023).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
