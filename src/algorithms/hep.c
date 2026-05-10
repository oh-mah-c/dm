#include "algorithms/hep.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint32_t *items;
    size_t length;
    uint32_t *tids;
    size_t num_tids;
} HEPItemset;

typedef struct {
    HEPItemset *itemsets;
    size_t count;
    size_t capacity;
} HEPItemsetList;

static void list_init(HEPItemsetList *list) {
    list->count = 0;
    list->capacity = 1024;
    list->itemsets = malloc(list->capacity * sizeof(HEPItemset));
}

static void list_add(HEPItemsetList *list, HEPItemset itemset) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->itemsets = realloc(list->itemsets, list->capacity * sizeof(HEPItemset));
    }
    list->itemsets[list->count++] = itemset;
}

static void list_free(HEPItemsetList *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->itemsets[i].items);
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

static bool check_ubo_and_o(uint32_t *tids, size_t size, uint32_t *g_tsize, size_t k, double xi, bool *is_ho) {
    if ((double)size < xi) return false;

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
    DM_HEP_Params *hep_params = (DM_HEP_Params *)params;
    double min_occ_param = hep_params ? hep_params->min_occupancy : 0.01;
    double xi = (min_occ_param < 1.0) ? (min_occ_param * ds->count) : min_occ_param;

    printf("[HEP] Starting on %zu transactions. Min Occupancy Threshold: %.2f\n", ds->count, xi);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    
    TransMeta *meta = malloc(ds->count * sizeof(TransMeta));
    for (size_t i = 0; i < ds->count; i++) {
        meta[i].orig_tid = i;
        meta[i].len = data[i].count;
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

    HEPItemsetList C_prev;
    list_init(&C_prev);
    
    size_t total_ho_count = 0;
    size_t total_ho_footprint = 0;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= xi) {
            bool is_ho = false;
            if (check_ubo_and_o(item_tids[i], counts[i], g_tsize, 1, xi, &is_ho)) {
                HEPItemset it;
                it.length = 1;
                it.items = malloc(sizeof(uint32_t));
                it.items[0] = i;
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
    while (C_prev.count > 0) {
        HEPItemsetList C_curr;
        list_init(&C_curr);
        
        for (size_t i = 0; i < C_prev.count; i++) {
            for (size_t j = i + 1; j < C_prev.count; j++) {
                HEPItemset *P1 = &C_prev.itemsets[i];
                HEPItemset *P2 = &C_prev.itemsets[j];
                
                bool match = true;
                for (size_t idx = 0; idx < k - 2; idx++) {
                    if (P1->items[idx] != P2->items[idx]) {
                        match = false;
                        break;
                    }
                }
                
                if (!match) break;
                
                uint32_t *new_tids = malloc((P1->num_tids < P2->num_tids ? P1->num_tids : P2->num_tids) * sizeof(uint32_t));
                size_t p1_idx = 0, p2_idx = 0, new_idx = 0;
                
                while (p1_idx < P1->num_tids && p2_idx < P2->num_tids) {
                    if (P1->tids[p1_idx] < P2->tids[p2_idx]) {
                        p1_idx++;
                    } else if (P1->tids[p1_idx] > P2->tids[p2_idx]) {
                        p2_idx++;
                    } else {
                        new_tids[new_idx++] = P1->tids[p1_idx];
                        p1_idx++;
                        p2_idx++;
                    }
                }
                
                if (new_idx >= xi) {
                    bool is_ho = false;
                    if (check_ubo_and_o(new_tids, new_idx, g_tsize, k, xi, &is_ho)) {
                        HEPItemset P;
                        P.length = k;
                        P.items = malloc(k * sizeof(uint32_t));
                        memcpy(P.items, P1->items, (k - 1) * sizeof(uint32_t));
                        P.items[k - 1] = P2->items[k - 2];
                        
                        P.num_tids = new_idx;
                        P.tids = malloc(new_idx * sizeof(uint32_t));
                        memcpy(P.tids, new_tids, new_idx * sizeof(uint32_t));
                        list_add(&C_curr, P);
                        
                        if (is_ho) {
                            total_ho_count++;
                            total_ho_footprint += k;
                        }
                    }
                }
                free(new_tids);
            }
        }
        
        list_free(&C_prev);
        C_prev = C_curr;
        k++;
    }

    free(g_tsize);
    printf("[HEP] Complete. Total high occupancy itemsets found: %zu\n", total_ho_count);
    dm_bench_record_results(total_ho_count, total_ho_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "hep",
    .name = "HEP Algorithm",
    .description = "High Efficient algorithm for mining high occupancy itemsets using UBO pruning.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
