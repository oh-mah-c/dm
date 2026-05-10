#include "algorithms/dic.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Dynamic Itemset Counting (DIC) Algorithm.
 * Reference: Sergey Brin, Rajeev Motwani, Jeffrey D. Ullman, Shalom Tsur, 
 * "Dynamic Itemset Counting and Implication Rules for Market Basket Data", in Proc. SIGMOD'97, 1997.
 */

typedef enum {
    DASHED_CIRCLE = 0, DASHED_SQUARE, SOLID_CIRCLE, SOLID_SQUARE
} DIC_State;

typedef struct DICNode {
    uint32_t item;
    uint32_t count;
    size_t start_proc; // total_processed when node was added
    DIC_State state;
    struct DICNode *children;
    struct DICNode *next_sibling;
} DICNode;

typedef struct {
    DICNode *root;
    uint32_t min_sup;
    uint32_t block_size;
    size_t N;
    size_t total_frequent;
    size_t total_footprint;
} DICContext;

typedef struct DICNodePool {
    DICNode *nodes;
    size_t capacity;
    size_t used;
    struct DICNodePool *next;
} DICNodePool;

static DICNodePool *g_pool = NULL;
static void pool_init() { g_pool = NULL; }
static DICNode* alloc_node(uint32_t item, size_t start_proc) {
    if (!g_pool || g_pool->used >= g_pool->capacity) {
        DICNodePool *p = malloc(sizeof(DICNodePool));
        p->capacity = 32768;
        p->nodes = calloc(p->capacity, sizeof(DICNode));
        p->used = 0; p->next = g_pool; g_pool = p;
    }
    DICNode *n = &g_pool->nodes[g_pool->used++];
    n->item = item; n->start_proc = start_proc; return n;
}
static void pool_free() {
    while (g_pool) {
        DICNodePool *next = g_pool->next;
        free(g_pool->nodes); free(g_pool); g_pool = next;
    }
    g_pool = NULL;
}

static DICNode* find_child(DICNode *parent, uint32_t item) {
    DICNode *curr = parent->children;
    while (curr) { if (curr->item == item) return curr; curr = curr->next_sibling; }
    return NULL;
}

static bool is_square(DICNode *n) { return n && (n->state == DASHED_SQUARE || n->state == SOLID_SQUARE); }

static void update_counters(DICNode *node, const uint32_t *items, uint32_t len, uint32_t idx) {
    if (node->item != 0xFFFFFFFF) {
        if (node->state == DASHED_CIRCLE || node->state == DASHED_SQUARE) node->count++;
    }
    DICNode *child = node->children;
    while (child) {
        for (uint32_t i = idx; i < len; i++) {
            if (items[i] == child->item) { update_counters(child, items, len, i + 1); break; }
            if (items[i] > child->item) break;
        }
        child = child->next_sibling;
    }
}

static bool check_subsets(DICContext *ctx, uint32_t *itemset, uint32_t k) {
    if (k <= 1) return true;
    for (uint32_t i = 0; i < k; i++) {
        DICNode *curr = ctx->root;
        bool found = true;
        for (uint32_t j = 0; j < k; j++) {
            if (i == j) continue;
            curr = find_child(curr, itemset[j]);
            if (!is_square(curr)) { found = false; break; }
        }
        if (!found) return false;
    }
    return true;
}

static void dic_step(DICNode *parent, uint32_t *prefix, uint32_t k, size_t total_proc, DICContext *ctx) {
    DICNode *child = parent->children;
    while (child) {
        if (child->state == DASHED_CIRCLE && child->count >= ctx->min_sup) child->state = DASHED_SQUARE;
        if ((child->state == DASHED_CIRCLE || child->state == DASHED_SQUARE) && (total_proc - child->start_proc) >= ctx->N) {
            child->state = (child->count >= ctx->min_sup) ? SOLID_SQUARE : SOLID_CIRCLE;
            if (child->state == SOLID_SQUARE) { ctx->total_frequent++; ctx->total_footprint += (k + 1); }
        }
        child = child->next_sibling;
    }

    DICNode *c1 = parent->children;
    while (c1) {
        if (is_square(c1)) {
            DICNode *c2 = parent->children;
            while (c2) {
                if (c2->item > c1->item && is_square(c2)) {
                    if (!find_child(c1, c2->item)) {
                        uint32_t *itemset = malloc((k + 2) * sizeof(uint32_t));
                        if (k > 0) memcpy(itemset, prefix, k * sizeof(uint32_t));
                        itemset[k] = c1->item; itemset[k + 1] = c2->item;
                        if (check_subsets(ctx, itemset, k + 2)) {
                            DICNode *n = alloc_node(c2->item, total_proc);
                            n->state = DASHED_CIRCLE;
                            n->next_sibling = c1->children; c1->children = n;
                        }
                        free(itemset);
                    }
                }
                c2 = c2->next_sibling;
            }
        }
        c1 = c1->next_sibling;
    }

    child = parent->children;
    uint32_t *new_prefix = malloc((k + 1) * sizeof(uint32_t));
    if (k > 0) memcpy(new_prefix, prefix, k * sizeof(uint32_t));
    while (child) {
        new_prefix[k] = child->item;
        dic_step(child, new_prefix, k + 1, total_proc, ctx);
        child = child->next_sibling;
    }
    free(new_prefix);
}

static bool has_dashed(DICNode *node) {
    if (node->state == DASHED_CIRCLE || node->state == DASHED_SQUARE) return true;
    DICNode *child = node->children;
    while (child) { if (has_dashed(child)) return true; child = child->next_sibling; }
    return false;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t ua = *(uint32_t*)a, ub = *(uint32_t*)b;
    return (ua < ub) ? -1 : (ua > ub ? 1 : 0);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_DIC_Params *p = (DM_DIC_Params*)params;
    double ms_val = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_val < 1.0) ? (uint32_t)ceil(ms_val * ds->count) : (uint32_t)ms_val;
    if (min_sup == 0) min_sup = 1;
    uint32_t M = (p && p->block_size > 0) ? p->block_size : 1000;

    printf("[DIC] Starting. Min Support: %u, Block Size: %u\n", min_sup, M);
    fflush(stdout);

    pool_init();
    DICNode *root = alloc_node(0xFFFFFFFF, 0);
    root->state = SOLID_SQUARE;

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple*)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
        qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_u32);
    }
    
    uint32_t num_items = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= 1) num_items++;
    uint32_t *items = malloc(num_items * sizeof(uint32_t));
    uint32_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= 1) items[idx++] = i;
    qsort(items, num_items, sizeof(uint32_t), cmp_u32);

    for (uint32_t i = 0; i < num_items; i++) {
        DICNode *n = alloc_node(items[i], 0);
        n->state = DASHED_CIRCLE;
        n->next_sibling = root->children; root->children = n;
    }

    DICContext ctx = { .root = root, .min_sup = min_sup, .block_size = M, .N = ds->count, 
                       .total_frequent = 0, .total_footprint = 0 };

    size_t current_tid = 0, total_processed = 0;
    while (has_dashed(root)) {
        for (uint32_t i = 0; i < M; i++) {
            update_counters(root, data[current_tid].items, data[current_tid].count, 0);
            current_tid = (current_tid + 1) % ds->count;
            total_processed++;
        }
        dic_step(root, NULL, 0, total_processed, &ctx);
        if (total_processed > ds->count * 500) break;
    }

    printf("[DIC] Complete. FIs found: %zu, Passes: %.2f\n", ctx.total_frequent, (double)total_processed / ds->count);
    dm_bench_record_results(ctx.total_frequent, ctx.total_footprint);
    free(counts); free(items); pool_free();
    return DM_SUCCESS;
}

static DM_Algorithm algo_dic = {
    .id = "dic", .name = "Dynamic Itemset Counting",
    .description = "An algorithm that counts itemsets dynamically during a pass, reducing total passes over data.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo_dic)
