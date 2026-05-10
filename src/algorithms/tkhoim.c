#include "algorithms/tkhoim.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    double occupancy;
    uint32_t *items;
    size_t length;
} TopKItem;

typedef struct {
    TopKItem *items;
    size_t count;
    size_t k;
} TopKList;

static void topk_init(TopKList *list, size_t k) {
    list->k = k;
    list->count = 0;
    list->items = malloc(k * sizeof(TopKItem));
}

static void topk_heapify_up(TopKList *list, size_t idx) {
    while (idx > 0) {
        size_t parent = (idx - 1) / 2;
        if (list->items[idx].occupancy < list->items[parent].occupancy) {
            TopKItem tmp = list->items[idx];
            list->items[idx] = list->items[parent];
            list->items[parent] = tmp;
            idx = parent;
        } else {
            break;
        }
    }
}

static void topk_heapify_down(TopKList *list, size_t idx) {
    while (true) {
        size_t left = 2 * idx + 1;
        size_t right = 2 * idx + 2;
        size_t smallest = idx;
        
        if (left < list->count && list->items[left].occupancy < list->items[smallest].occupancy) {
            smallest = left;
        }
        if (right < list->count && list->items[right].occupancy < list->items[smallest].occupancy) {
            smallest = right;
        }
        
        if (smallest != idx) {
            TopKItem tmp = list->items[idx];
            list->items[idx] = list->items[smallest];
            list->items[smallest] = tmp;
            idx = smallest;
        } else {
            break;
        }
    }
}

static void topk_insert(TopKList *list, double occ, uint32_t *items, size_t length) {
    if (list->count < list->k) {
        list->items[list->count].occupancy = occ;
        list->items[list->count].length = length;
        list->items[list->count].items = malloc(length * sizeof(uint32_t));
        memcpy(list->items[list->count].items, items, length * sizeof(uint32_t));
        topk_heapify_up(list, list->count);
        list->count++;
    } else {
        if (occ > list->items[0].occupancy) {
            free(list->items[0].items);
            list->items[0].occupancy = occ;
            list->items[0].length = length;
            list->items[0].items = malloc(length * sizeof(uint32_t));
            memcpy(list->items[0].items, items, length * sizeof(uint32_t));
            topk_heapify_down(list, 0);
        }
    }
}

static double topk_get_minO(TopKList *list) {
    if (list->count < list->k) return 0.0;
    return list->items[0].occupancy;
}

typedef struct {
    uint32_t item;
    uint32_t *tids;
    size_t num_tids;
    double ubo;
    uint32_t l1;
} TKHOIMItemset;

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

static double calc_ubo(uint32_t *tids, size_t size, uint32_t *g_tsize) {
    double current_sum = 0.0;
    double max_ubo = 0.0;
    for (int i = (int)size - 1; i >= 0; i--) {
        uint32_t tsize = g_tsize[tids[i]];
        current_sum += 1.0 / (double)tsize;
        if (i == 0 || g_tsize[tids[i-1]] < tsize) {
            double current_ubo = tsize * current_sum;
            if (current_ubo > max_ubo) max_ubo = current_ubo;
        }
    }
    return max_ubo;
}

static double calc_lubo(TKHOIMItemset *Y, size_t len_X, size_t nre_X) {
    return (double)Y->num_tids * (double)(len_X + nre_X) / (double)Y->l1;
}

static void search_tkhoim(TopKList *CI, TKHOIMItemset *PI, size_t pi_size, size_t prefix_len, uint32_t *prefix, uint32_t *g_tsize, uint32_t *buffer, bool hastheSameLength) {
    for (size_t i = 0; i < pi_size; i++) {
        TKHOIMItemset *X = &PI[i];
        size_t nre_X = pi_size - i - 1;
        double minO = topk_get_minO(CI);
        
        if (X->ubo < minO) continue;
        if (calc_lubo(X, prefix_len + 1, nre_X) < minO) continue;
        
        TKHOIMItemset *new_PI = NULL;
        if (nre_X > 0) {
            new_PI = malloc(nre_X * sizeof(TKHOIMItemset));
        }
        size_t new_pi_size = 0;
        
        for (size_t j = i + 1; j < pi_size; j++) {
            TKHOIMItemset *Y = &PI[j];
            minO = topk_get_minO(CI);
            
            if (Y->ubo < minO) {
                nre_X--;
                continue;
            }
            if (calc_lubo(Y, prefix_len + 1, nre_X) < minO) {
                nre_X--;
                continue;
            }
            
            size_t p1_idx = 0, p2_idx = 0, new_idx = 0;
            while (p1_idx < X->num_tids && p2_idx < Y->num_tids) {
                if (X->tids[p1_idx] < Y->tids[p2_idx]) {
                    p1_idx++;
                } else if (X->tids[p1_idx] > Y->tids[p2_idx]) {
                    p2_idx++;
                } else {
                    buffer[new_idx++] = X->tids[p1_idx];
                    p1_idx++;
                    p2_idx++;
                }
            }
            
            if (new_idx == 0) {
                nre_X--;
                continue;
            }
            
            double ubo_XY = hastheSameLength ? (double)new_idx : calc_ubo(buffer, new_idx, g_tsize);
            if (ubo_XY < minO) {
                nre_X--;
                continue;
            }
            
            TKHOIMItemset *new_itemset = &new_PI[new_pi_size++];
            new_itemset->item = Y->item;
            new_itemset->num_tids = new_idx;
            new_itemset->tids = malloc(new_idx * sizeof(uint32_t));
            memcpy(new_itemset->tids, buffer, new_idx * sizeof(uint32_t));
            new_itemset->ubo = ubo_XY;
            new_itemset->l1 = g_tsize[buffer[0]];
            
            double sum_inv_len = 0.0;
            if (hastheSameLength) {
                sum_inv_len = (double)new_idx / (double)new_itemset->l1;
            } else {
                for (size_t t = 0; t < new_idx; t++) {
                    sum_inv_len += 1.0 / (double)g_tsize[buffer[t]];
                }
            }
            double occ_XY = (prefix_len + 2) * sum_inv_len;
            
            if (occ_XY >= minO) {
                prefix[prefix_len + 1] = Y->item;
                prefix[prefix_len] = X->item; // Ensure X is in prefix
                topk_insert(CI, occ_XY, prefix, prefix_len + 2);
                minO = topk_get_minO(CI);
            }
        }
        
        if (new_pi_size > 0) {
            prefix[prefix_len] = X->item;
            search_tkhoim(CI, new_PI, new_pi_size, prefix_len + 1, prefix, g_tsize, buffer, hastheSameLength);
        }
        
        if (new_PI) {
            for (size_t c = 0; c < new_pi_size; c++) {
                free(new_PI[c].tids);
            }
            free(new_PI);
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_TKHOIM_Params *tkhoim_params = (DM_TKHOIM_Params *)params;
    size_t k = tkhoim_params ? tkhoim_params->k : 10;
    if (k == 0) k = 10;

    printf("[TKHOIM] Starting on %zu transactions. Top-K: %zu\n", ds->count, k);

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
        if (counts[i] > 0) {
            item_tids[i] = malloc(counts[i] * sizeof(uint32_t));
        } else {
            item_tids[i] = NULL;
        }
    }

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < data[orig].count; j++) {
            uint32_t item = data[orig].items[j];
            item_tids[item][item_idx[item]++] = i;
        }
    }
    free(meta);
    free(item_idx);

    TopKList CI;
    topk_init(&CI, k);
    
    size_t nre_root = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] > 0) nre_root++;
    }
    
    TKHOIMItemset *PI = malloc(nre_root * sizeof(TKHOIMItemset));
    size_t pi_size = 0;
    
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] == 0) continue;
        
        double minO = topk_get_minO(&CI);
        uint32_t *tids = item_tids[i];
        size_t num_tids = counts[i];
        
        double ubo = hastheSameLength ? (double)num_tids : calc_ubo(tids, num_tids, g_tsize);
        if (ubo < minO) {
            nre_root--;
            continue;
        }
        
        uint32_t l1 = g_tsize[tids[0]];
        double lubo = (double)num_tids * (double)(0 + nre_root) / (double)l1;
        if (lubo < minO) {
            nre_root--;
            continue;
        }
        
        double sum_inv_len = 0.0;
        if (hastheSameLength) {
            sum_inv_len = (double)num_tids / (double)l1;
        } else {
            for (size_t t = 0; t < num_tids; t++) {
                sum_inv_len += 1.0 / (double)g_tsize[tids[t]];
            }
        }
        double occ = 1.0 * sum_inv_len;
        
        TKHOIMItemset *new_itemset = &PI[pi_size++];
        new_itemset->item = i;
        new_itemset->num_tids = num_tids;
        new_itemset->tids = malloc(num_tids * sizeof(uint32_t));
        memcpy(new_itemset->tids, tids, num_tids * sizeof(uint32_t));
        new_itemset->ubo = ubo;
        new_itemset->l1 = l1;
        
        if (occ >= minO) {
            uint32_t item_buf = i;
            topk_insert(&CI, occ, &item_buf, 1);
        }
    }
    
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_tids[i]) free(item_tids[i]);
    }
    free(item_tids);
    free(counts);

    uint32_t *buffer = malloc(ds->count * sizeof(uint32_t));
    uint32_t *prefix = malloc((ds->max_id + 1) * sizeof(uint32_t));
    
    search_tkhoim(&CI, PI, pi_size, 0, prefix, g_tsize, buffer, hastheSameLength);
    
    free(buffer);
    free(prefix);
    
    for (size_t i = 0; i < pi_size; i++) {
        free(PI[i].tids);
    }
    free(PI);
    free(g_tsize);
    
    size_t total_ho_count = CI.count;
    size_t total_ho_footprint = 0;
    
    for (size_t i = 0; i < CI.count; i++) {
        total_ho_footprint += CI.items[i].length;
        free(CI.items[i].items);
    }
    free(CI.items);

    printf("[TKHOIM] Complete. Total top-k itemsets found: %zu\n", total_ho_count);
    dm_bench_record_results(total_ho_count, total_ho_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "tkhoim",
    .name = "TKHOIM Algorithm",
    .description = "Top-k High Occupancy Itemset Miner (Yildirim 2025).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
