#include "algorithms/charm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HASH_TABLE_SIZE 1000003

typedef struct CFI_Node {
    uint32_t *items;
    size_t num_items;
    uint32_t support;
    uint32_t tid_sum;
    struct CFI_Node *next;
} CFI_Node;

typedef struct {
    uint32_t *items;
    size_t num_items;
    uint32_t *tids;
    size_t num_tids;
    bool deleted;
} CharmNode;

typedef struct {
    CFI_Node **hash_table;
    size_t total_fci;
    size_t total_footprint;
    uint32_t min_sup;
    uint32_t max_item_id;
    uint32_t *pair_counts;
    uint32_t *item_to_idx;
    bool is_root;
    size_t freq_count;
} CharmContext;

static int cmp_uint32(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    if (ua < ub) return -1;
    if (ua > ub) return 1;
    return 0;
}

static int cmp_node_supp_asc(const void *a, const void *b) {
    const CharmNode *na = (const CharmNode *)a;
    const CharmNode *nb = (const CharmNode *)b;
    if (na->num_tids < nb->num_tids) return -1;
    if (na->num_tids > nb->num_tids) return 1;
    if (na->items[0] < nb->items[0]) return -1;
    if (na->items[0] > nb->items[0]) return 1;
    return 0;
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

static bool is_subset(const uint32_t *a, size_t len_a, const uint32_t *b, size_t len_b) {
    if (len_a > len_b) return false;
    size_t i = 0, j = 0;
    while (i < len_a && j < len_b) {
        if (a[i] < b[j]) return false;
        else if (a[i] == b[j]) { i++; j++; }
        else j++;
    }
    return i == len_a;
}

static bool is_subsumed(CharmContext *ctx, const uint32_t *items, size_t num_items, uint32_t supp, uint32_t tid_sum) {
    uint32_t hash_idx = tid_sum % HASH_TABLE_SIZE;
    for (CFI_Node *n = ctx->hash_table[hash_idx]; n != NULL; n = n->next) {
        if (n->support == supp && n->tid_sum == tid_sum) {
            if (is_subset(items, num_items, n->items, n->num_items)) {
                return true;
            }
        }
    }
    return false;
}

static void add_cfi(CharmContext *ctx, const uint32_t *items, size_t num_items, uint32_t supp, uint32_t tid_sum) {
    uint32_t hash_idx = tid_sum % HASH_TABLE_SIZE;
    CFI_Node *n = malloc(sizeof(CFI_Node));
    n->items = malloc(num_items * sizeof(uint32_t));
    memcpy(n->items, items, num_items * sizeof(uint32_t));
    n->num_items = num_items;
    n->support = supp;
    n->tid_sum = tid_sum;
    n->next = ctx->hash_table[hash_idx];
    ctx->hash_table[hash_idx] = n;
    
    ctx->total_fci++;
    ctx->total_footprint += num_items;
}

static size_t intersect_tids(const uint32_t *tids1, size_t len1, const uint32_t *tids2, size_t len2, uint32_t *out) {
    if (tids1 == tids2) {
        memcpy(out, tids1, len1 * sizeof(uint32_t));
        return len1;
    }
    size_t i = 0, j = 0, k = 0;
    while (i < len1 && j < len2) {
        if (tids1[i] < tids2[j]) i++;
        else if (tids1[i] > tids2[j]) j++;
        else {
            out[k++] = tids1[i];
            i++; j++;
        }
    }
    return k;
}

static void charm_extend(CharmNode *nodes, size_t num_nodes, const uint32_t *prefix, size_t prefix_len, CharmContext *ctx) {
    for (size_t i = 0; i < num_nodes; i++) {
        if (nodes[i].deleted) continue;
        
        uint32_t *curr_prefix = malloc(ctx->freq_count * sizeof(uint32_t));
        if (prefix_len > 0) {
            memcpy(curr_prefix, prefix, prefix_len * sizeof(uint32_t));
        }
        memcpy(curr_prefix + prefix_len, nodes[i].items, nodes[i].num_items * sizeof(uint32_t));
        size_t curr_len = prefix_len + nodes[i].num_items;
        
        CharmNode *NewN = malloc(num_nodes * sizeof(CharmNode));
        size_t new_count = 0;
        
        for (size_t j = i + 1; j < num_nodes; j++) {
            if (nodes[j].deleted) continue;
            
            if (ctx->is_root && ctx->pair_counts) {
                uint32_t item_i = nodes[i].items[0];
                uint32_t item_j = nodes[j].items[0];
                uint32_t idx_i = ctx->item_to_idx[item_i];
                uint32_t idx_j = ctx->item_to_idx[item_j];
                size_t p_idx = (idx_i > idx_j) ? ((size_t)idx_i * (idx_i - 1) / 2 + idx_j) : ((size_t)idx_j * (idx_j - 1) / 2 + idx_i);
                if (ctx->pair_counts[p_idx] < ctx->min_sup) continue;
            }
            
            size_t min_len = nodes[i].num_tids < nodes[j].num_tids ? nodes[i].num_tids : nodes[j].num_tids;
            uint32_t *t_new = malloc(min_len * sizeof(uint32_t));
            size_t t_new_len = intersect_tids(nodes[i].tids, nodes[i].num_tids, nodes[j].tids, nodes[j].num_tids, t_new);
            
            if (t_new_len >= ctx->min_sup) {
                bool subset1 = (t_new_len == nodes[i].num_tids); // t_i \subset t_j
                bool subset2 = (t_new_len == nodes[j].num_tids); // t_i \supset t_j
                
                if (subset1 && subset2) { // Prop 1
                    memcpy(curr_prefix + curr_len, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    curr_len += nodes[j].num_items;
                    nodes[j].deleted = true;
                    free(t_new);
                } else if (subset1) { // Prop 2
                    memcpy(curr_prefix + curr_len, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    curr_len += nodes[j].num_items;
                    free(t_new);
                } else if (subset2) { // Prop 3
                    NewN[new_count].items = malloc(nodes[j].num_items * sizeof(uint32_t));
                    memcpy(NewN[new_count].items, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    NewN[new_count].num_items = nodes[j].num_items;
                    NewN[new_count].tids = t_new;
                    NewN[new_count].num_tids = t_new_len;
                    NewN[new_count].deleted = false;
                    new_count++;
                    nodes[j].deleted = true;
                } else { // Prop 4
                    NewN[new_count].items = malloc(nodes[j].num_items * sizeof(uint32_t));
                    memcpy(NewN[new_count].items, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    NewN[new_count].num_items = nodes[j].num_items;
                    NewN[new_count].tids = t_new;
                    NewN[new_count].num_tids = t_new_len;
                    NewN[new_count].deleted = false;
                    new_count++;
                }
            } else {
                free(t_new);
            }
        }
        
        if (new_count > 0) {
            qsort(NewN, new_count, sizeof(CharmNode), cmp_node_supp_asc);
            bool was_root = ctx->is_root;
            ctx->is_root = false;
            charm_extend(NewN, new_count, curr_prefix, curr_len, ctx);
            ctx->is_root = was_root;
            
            for (size_t x = 0; x < new_count; x++) {
                free(NewN[x].items);
                free(NewN[x].tids);
            }
        }
        free(NewN);
        
        uint32_t sum = 0;
        for (size_t x = 0; x < nodes[i].num_tids; x++) {
            sum += nodes[i].tids[x];
        }
        
        qsort(curr_prefix, curr_len, sizeof(uint32_t), cmp_uint32);
        
        if (!is_subsumed(ctx, curr_prefix, curr_len, nodes[i].num_tids, sum)) {
            add_cfi(ctx, curr_prefix, curr_len, nodes[i].num_tids, sum);
        }
        
        free(curr_prefix);
    }
}

static void free_hash_table(CharmContext *ctx) {
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        CFI_Node *n = ctx->hash_table[i];
        while (n != NULL) {
            CFI_Node *next = n->next;
            free(n->items);
            free(n);
            n = next;
        }
    }
    free(ctx->hash_table);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_CHARM_Params *charm_params = (DM_CHARM_Params *)params;
    double min_sup_param = charm_params ? charm_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;
    
    printf("[CHARM] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);
    
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }
    
    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }
    
    if (freq_count == 0) {
        printf("[CHARM] Complete. Total frequent closed itemsets found: 0\n");
        free(counts);
        return DM_SUCCESS;
    }
    
    uint32_t *freq_items = malloc(freq_count * sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_items[idx++] = i;
    }
    
    g_counts = counts;
    qsort(freq_items, freq_count, sizeof(uint32_t), cmp_freq_asc);
    
    uint32_t *item_to_idx = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < freq_count; i++) {
        item_to_idx[freq_items[i]] = (uint32_t)i;
    }
    
    size_t pair_arr_size = (size_t)freq_count * (freq_count - 1) / 2;
    uint32_t *pair_counts = pair_arr_size > 0 ? calloc(pair_arr_size, sizeof(uint32_t)) : NULL;
    
    if (pair_counts) {
        for (size_t i = 0; i < ds->count; i++) {
            uint32_t *filtered = malloc(data[i].count * sizeof(uint32_t));
            size_t f_count = 0;
            for (size_t j = 0; j < data[i].count; j++) {
                if (counts[data[i].items[j]] >= min_sup) {
                    filtered[f_count++] = data[i].items[j];
                }
            }
            if (f_count > 1) {
                for (size_t x = 0; x < f_count; x++) {
                    uint32_t idx_x = item_to_idx[filtered[x]];
                    for (size_t y = 0; y < x; y++) {
                        uint32_t idx_y = item_to_idx[filtered[y]];
                        size_t p_idx = (idx_x > idx_y) ? ((size_t)idx_x * (idx_x - 1) / 2 + idx_y) : ((size_t)idx_y * (idx_y - 1) / 2 + idx_x);
                        pair_counts[p_idx]++;
                    }
                }
            }
            free(filtered);
        }
    }
    
    CharmNode *root_nodes = calloc(freq_count, sizeof(CharmNode));
    for (size_t i = 0; i < freq_count; i++) {
        root_nodes[i].items = malloc(sizeof(uint32_t));
        root_nodes[i].items[0] = freq_items[i];
        root_nodes[i].num_items = 1;
        root_nodes[i].tids = malloc(counts[freq_items[i]] * sizeof(uint32_t));
        root_nodes[i].num_tids = 0;
        root_nodes[i].deleted = false;
    }
    
    for (size_t tid = 0; tid < ds->count; tid++) {
        for (size_t j = 0; j < data[tid].count; j++) {
            uint32_t item = data[tid].items[j];
            if (counts[item] >= min_sup) {
                uint32_t curr_idx = item_to_idx[item];
                root_nodes[curr_idx].tids[root_nodes[curr_idx].num_tids++] = (uint32_t)tid;
            }
        }
    }
    
    CharmContext char_ctx;
    char_ctx.hash_table = calloc(HASH_TABLE_SIZE, sizeof(CFI_Node*));
    char_ctx.total_fci = 0;
    char_ctx.total_footprint = 0;
    char_ctx.min_sup = min_sup;
    char_ctx.max_item_id = ds->max_id;
    char_ctx.pair_counts = pair_counts;
    char_ctx.item_to_idx = item_to_idx;
    char_ctx.is_root = true;
    char_ctx.freq_count = freq_count;
    
    charm_extend(root_nodes, freq_count, NULL, 0, &char_ctx);
    
    printf("[CHARM] Complete. Total frequent closed itemsets found: %zu\n", char_ctx.total_fci);
    dm_bench_record_results(char_ctx.total_fci, char_ctx.total_footprint);
    
    for (size_t i = 0; i < freq_count; i++) {
        free(root_nodes[i].items);
        free(root_nodes[i].tids);
    }
    free(root_nodes);
    free(counts);
    free(freq_items);
    free(item_to_idx);
    if (pair_counts) free(pair_counts);
    free_hash_table(&char_ctx);
    
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "charm",
    .name = "CHARM Algorithm",
    .description = "Mining Closed Association Rules using itemset-tidset pairs (Zaki & Hsiao 2002).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
