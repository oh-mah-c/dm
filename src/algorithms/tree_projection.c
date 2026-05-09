#include "algorithms/tree_projection.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint32_t *items;
    size_t count;
} TP_Trans;

typedef struct {
    TP_Trans *trans;
    size_t count;
} TP_TransSet;

typedef struct {
    uint32_t min_sup;
    size_t total_fi;
    size_t total_footprint;
} TP_Context;

static void tree_project(uint32_t *prefix, size_t prefix_len, 
                         uint32_t *E, uint32_t *E_sup, size_t E_len, 
                         TP_TransSet *T, TP_Context *ctx) {
    if (E_len == 0) return;
    
    size_t mat_size = (E_len * (E_len - 1)) / 2;
    uint32_t *M = NULL;
    if (mat_size > 0) {
        M = calloc(mat_size, sizeof(uint32_t));
        for (size_t t = 0; t < T->count; t++) {
            uint32_t *items = T->trans[t].items;
            size_t len = T->trans[t].count;
            for (size_t i_idx = 0; i_idx < len; i_idx++) {
                uint32_t a = items[i_idx];
                size_t base = a * E_len - (a * (a + 1)) / 2 - a - 1;
                for (size_t j_idx = i_idx + 1; j_idx < len; j_idx++) {
                    uint32_t b = items[j_idx];
                    M[base + b]++;
                }
            }
        }
    }
    
    for (size_t i = 0; i < E_len; i++) {
        // Record frequent itemset P U {E[i]}
        ctx->total_fi++;
        ctx->total_footprint += prefix_len + 1;
        
        uint32_t *E_new = NULL;
        uint32_t *E_new_sup = NULL;
        uint32_t *mapping = NULL;
        size_t E_new_len = 0;
        
        if (i + 1 < E_len) {
            mapping = malloc(E_len * sizeof(uint32_t));
            E_new = malloc((E_len - i - 1) * sizeof(uint32_t));
            E_new_sup = malloc((E_len - i - 1) * sizeof(uint32_t));
            
            size_t base = i * E_len - (i * (i + 1)) / 2 - i - 1;
            for (size_t j = i + 1; j < E_len; j++) {
                uint32_t pair_sup = M[base + j];
                if (pair_sup >= ctx->min_sup) {
                    mapping[j] = E_new_len;
                    E_new[E_new_len] = E[j];
                    E_new_sup[E_new_len] = pair_sup;
                    E_new_len++;
                }
            }
        }
        
        if (E_new_len > 0) {
            TP_TransSet T_new;
            T_new.trans = malloc(T->count * sizeof(TP_Trans));
            T_new.count = 0;
            
            size_t base = i * E_len - (i * (i + 1)) / 2 - i - 1;
            
            for (size_t t = 0; t < T->count; t++) {
                uint32_t *items = T->trans[t].items;
                size_t len = T->trans[t].count;
                
                bool has_i = false;
                size_t idx_i = 0;
                for (size_t k = 0; k < len; k++) {
                    if (items[k] == i) {
                        has_i = true;
                        idx_i = k;
                        break;
                    }
                    if (items[k] > i) break;
                }
                
                if (has_i && idx_i + 1 < len) {
                    uint32_t *new_items = malloc((len - idx_i - 1) * sizeof(uint32_t));
                    size_t new_len = 0;
                    for (size_t k = idx_i + 1; k < len; k++) {
                        uint32_t item_j = items[k];
                        if (M[base + item_j] >= ctx->min_sup) {
                            new_items[new_len++] = mapping[item_j];
                        }
                    }
                    
                    if (new_len >= 2) {
                        T_new.trans[T_new.count].items = new_items;
                        T_new.trans[T_new.count].count = new_len;
                        T_new.count++;
                    } else {
                        free(new_items);
                    }
                }
            }
            
            uint32_t *new_prefix = malloc((prefix_len + 1) * sizeof(uint32_t));
            if (prefix_len > 0) memcpy(new_prefix, prefix, prefix_len * sizeof(uint32_t));
            new_prefix[prefix_len] = E[i];
            
            tree_project(new_prefix, prefix_len + 1, E_new, E_new_sup, E_new_len, &T_new, ctx);
            
            for (size_t t = 0; t < T_new.count; t++) {
                free(T_new.trans[t].items);
            }
            free(T_new.trans);
            free(new_prefix);
        }
        
        if (E_new) free(E_new);
        if (E_new_sup) free(E_new_sup);
        if (mapping) free(mapping);
    }
    
    if (M) free(M);
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

static int cmp_uint32(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_TREE_PROJECTION_Params *tp_params = (DM_TREE_PROJECTION_Params *)params;
    double min_sup_param = tp_params ? tp_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[TreeProjection] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    uint32_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        printf("[TreeProjection] Complete. Total frequent itemsets found: 0\n");
        free(counts);
        return DM_SUCCESS;
    }

    uint32_t *L1 = malloc(freq_count * sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) L1[idx++] = i;
    }
    
    g_counts = counts;
    qsort(L1, freq_count, sizeof(uint32_t), cmp_freq_asc);
    
    uint32_t *L1_sup = malloc(freq_count * sizeof(uint32_t));
    for (size_t i = 0; i < freq_count; i++) {
        L1_sup[i] = counts[L1[i]];
    }

    uint32_t *global_to_local = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) global_to_local[i] = UINT32_MAX;
    for (uint32_t i = 0; i < freq_count; i++) {
        global_to_local[L1[i]] = i;
    }

    TP_TransSet T_root;
    T_root.trans = malloc(ds->count * sizeof(TP_Trans));
    T_root.count = 0;

    for (size_t t = 0; t < ds->count; t++) {
        size_t len = data[t].count;
        uint32_t *local_items = malloc(len * sizeof(uint32_t));
        size_t local_len = 0;
        
        for (size_t i = 0; i < len; i++) {
            uint32_t global_item = data[t].items[i];
            uint32_t local_idx = global_to_local[global_item];
            if (local_idx != UINT32_MAX) {
                local_items[local_len++] = local_idx;
            }
        }
        
        if (local_len >= 2) {
            qsort(local_items, local_len, sizeof(uint32_t), cmp_uint32);
            T_root.trans[T_root.count].items = local_items;
            T_root.trans[T_root.count].count = local_len;
            T_root.count++;
        } else {
            free(local_items);
        }
    }

    TP_Context ctx = { min_sup, 0, 0 };
    
    tree_project(NULL, 0, L1, L1_sup, freq_count, &T_root, &ctx);

    printf("[TreeProjection] Complete. Total frequent itemsets found: %zu\n", ctx.total_fi);
    dm_bench_record_results(ctx.total_fi, ctx.total_footprint);

    for (size_t t = 0; t < T_root.count; t++) {
        free(T_root.trans[t].items);
    }
    free(T_root.trans);
    free(global_to_local);
    free(L1);
    free(L1_sup);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "tree_projection",
    .name = "TreeProjection Algorithm",
    .description = "A Tree Projection Algorithm for Generation of Frequent Itemsets using Matrix Counting.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
