#include "algorithms/lcm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint32_t min_sup;
    uint32_t max_id; // Mapped max id
    uint32_t *inv_map; // Mapped ID -> Original ID
    size_t total_closed;
    size_t total_footprint;
} LCMContext;

static void lcm_closed(uint32_t *P, size_t P_len, int32_t core_i, DM_Trans_Simple *db, size_t db_size, LCMContext *ctx) {
    if (db_size < ctx->min_sup) return;

    uint32_t *counts = calloc(ctx->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < db_size; i++) {
        for (size_t j = 0; j < db[i].count; j++) {
            counts[db[i].items[j]]++;
        }
    }

    size_t total_bucket_elements = 0;
    for (int32_t e = core_i + 1; e <= (int32_t)ctx->max_id; e++) {
        if (counts[e] >= ctx->min_sup) total_bucket_elements += counts[e];
    }

    if (total_bucket_elements == 0) {
        free(counts);
        return;
    }

    uint32_t **buckets = malloc((ctx->max_id + 1) * sizeof(uint32_t *));
    size_t *bucket_counts = calloc(ctx->max_id + 1, sizeof(size_t));
    uint32_t *flat_buckets = malloc(total_bucket_elements * sizeof(uint32_t));
    
    size_t flat_b_idx = 0;
    for (int32_t e = core_i + 1; e <= (int32_t)ctx->max_id; e++) {
        if (counts[e] >= ctx->min_sup) {
            buckets[e] = &flat_buckets[flat_b_idx];
            flat_b_idx += counts[e];
        } else {
            buckets[e] = NULL;
        }
    }

    for (size_t i = 0; i < db_size; i++) {
        for (size_t j = 0; j < db[i].count; j++) {
            int32_t e = db[i].items[j];
            if (e > core_i && buckets[e] != NULL) {
                buckets[e][bucket_counts[e]++] = i;
            }
        }
    }

    for (int32_t e = core_i + 1; e <= (int32_t)ctx->max_id; e++) {
        if (buckets[e] == NULL) continue;
        
        uint32_t *tids = buckets[e];
        size_t tids_len = bucket_counts[e];
        
        uint32_t *new_counts = calloc(ctx->max_id + 1, sizeof(uint32_t));
        size_t total_items_in_tids = 0;
        for (size_t i = 0; i < tids_len; i++) {
            uint32_t tid = tids[i];
            total_items_in_tids += db[tid].count;
            for (size_t j = 0; j < db[tid].count; j++) {
                new_counts[db[tid].items[j]]++;
            }
        }
        
        bool is_ppc = true;
        for (int32_t x = 0; x < e; x++) {
            if (new_counts[x] == tids_len) {
                is_ppc = false;
                break;
            }
        }
        
        if (is_ppc) {
            size_t add_len = 0;
            for (int32_t y = e; y <= (int32_t)ctx->max_id; y++) {
                if (new_counts[y] == tids_len) add_len++;
            }
            
            uint32_t *P_prime = malloc((P_len + add_len) * sizeof(uint32_t));
            if (P_len > 0) memcpy(P_prime, P, P_len * sizeof(uint32_t));
            size_t idx = P_len;
            for (int32_t y = e; y <= (int32_t)ctx->max_id; y++) {
                if (new_counts[y] == tids_len) P_prime[idx++] = ctx->inv_map[y];
            }
            
            ctx->total_closed++;
            ctx->total_footprint += P_len + add_len;
            
            DM_Trans_Simple *new_db = malloc(tids_len * sizeof(DM_Trans_Simple));
            uint32_t *flat_items = malloc(total_items_in_tids * sizeof(uint32_t));
            size_t flat_idx = 0;
            
            for (size_t i = 0; i < tids_len; i++) {
                uint32_t tid = tids[i];
                new_db[i].items = &flat_items[flat_idx];
                size_t new_len = 0;
                for (size_t j = 0; j < db[tid].count; j++) {
                    uint32_t item = db[tid].items[j];
                    if (new_counts[item] >= ctx->min_sup && new_counts[item] < tids_len) {
                        new_db[i].items[new_len++] = item;
                    }
                }
                new_db[i].count = new_len;
                flat_idx += new_len;
            }
            
            lcm_closed(P_prime, P_len + add_len, e, new_db, tids_len, ctx);
            
            free(flat_items);
            free(new_db);
            free(P_prime);
        }
        
        free(new_counts);
    }
    
    free(flat_buckets);
    free(bucket_counts);
    free(buckets);
    free(counts);
}

static uint32_t *g_counts = NULL;
static int cmp_freq_asc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (g_counts[ia] < g_counts[ib]) return -1;
    if (g_counts[ia] > g_counts[ib]) return 1;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_LCM_Params *lcm_params = (DM_LCM_Params *)params;
    double min_sup_param = lcm_params ? lcm_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[LCM] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }
    
    g_counts = counts;

    size_t initial_tids_len = ds->count;
    size_t P_len = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] == initial_tids_len) P_len++;
    }
    
    uint32_t *P = NULL;
    if (P_len > 0) {
        P = malloc(P_len * sizeof(uint32_t));
        size_t idx = 0;
        for (uint32_t i = 0; i <= ds->max_id; i++) {
            if (counts[i] == initial_tids_len) P[idx++] = i;
        }
    }

    uint32_t *freq_items = malloc((ds->max_id + 1) * sizeof(uint32_t));
    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup && counts[i] < initial_tids_len) {
            freq_items[freq_count++] = i;
        }
    }

    if (freq_count == 0) {
        printf("[LCM] Complete. Total frequent closed itemsets found: 1\n");
        dm_bench_record_results(1, P_len);
        if (P) free(P);
        free(freq_items);
        free(counts);
        return DM_SUCCESS;
    }

    qsort(freq_items, freq_count, sizeof(uint32_t), cmp_freq_asc);

    uint32_t *item_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    uint32_t *inv_map = malloc(freq_count * sizeof(uint32_t));
    memset(item_map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    
    for (size_t i = 0; i < freq_count; i++) {
        item_map[freq_items[i]] = i;
        inv_map[i] = freq_items[i];
    }

    LCMContext ctx = {0};
    ctx.min_sup = min_sup;
    ctx.max_id = freq_count - 1;
    ctx.inv_map = inv_map;
    ctx.total_closed = 1; // P itself is a closed itemset
    ctx.total_footprint = P_len;

    DM_Trans_Simple *new_db = malloc(initial_tids_len * sizeof(DM_Trans_Simple));
    size_t total_items_in_db = 0;
    for (size_t i = 0; i < initial_tids_len; i++) {
        total_items_in_db += data[i].count;
    }
    
    uint32_t *flat_items = malloc(total_items_in_db * sizeof(uint32_t));
    size_t flat_idx = 0;
    for (size_t i = 0; i < initial_tids_len; i++) {
        new_db[i].items = &flat_items[flat_idx];
        size_t new_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup && counts[item] < initial_tids_len) {
                new_db[i].items[new_len++] = item_map[item];
            }
        }
        new_db[i].count = new_len;
        flat_idx += new_len;
    }

    lcm_closed(P, P_len, -1, new_db, initial_tids_len, &ctx);

    printf("[LCM] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_closed);
    dm_bench_record_results(ctx.total_closed, ctx.total_footprint);

    free(flat_items);
    free(new_db);
    free(item_map);
    free(inv_map);
    free(freq_items);
    if (P) free(P);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "lcm",
    .name = "LCM Algorithm",
    .description = "Linear time Closed itemset Miner using Prefix Preserving Closure Extension.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
