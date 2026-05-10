#include "algorithms/talky_g.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Talky-G Algorithm for vertical mining of frequent generators.
 * Reference: Laszlo Szathmary et al., "Efficient Vertical Mining of Frequent Closures and Generators", IDA 2009.
 */

#define HASH_SIZE 65536

typedef struct GenNode {
    uint32_t *items;
    size_t items_len;
    uint32_t support;
    uint32_t tid_sum;
    struct GenNode *next;
} GenNode;

typedef struct {
    uint32_t *items;
    size_t items_len;
    uint32_t *tids;
    size_t tids_len;
    uint32_t support;
    uint32_t tid_sum;
} ITNode;

typedef struct {
    GenNode **buckets;
    uint32_t min_sup;
    size_t total_fgs;
    size_t total_footprint;
    size_t num_rows;
} TalkyContext;

// Subset check: is a \subset b? (Proper subset)
static bool is_subset(const uint32_t *a, size_t len_a, const uint32_t *b, size_t len_b) {
    if (len_a >= len_b) return false;
    size_t i = 0, j = 0;
    while (i < len_a && j < len_b) {
        if (a[i] < b[j]) return false;
        if (a[i] == b[j]) i++;
        j++;
    }
    return i == len_a;
}

static bool test3_subsumption(TalkyContext *ctx, const uint32_t *items, size_t items_len, uint32_t support, uint32_t tid_sum) {
    uint32_t idx = tid_sum % HASH_SIZE;
    for (GenNode *curr = ctx->buckets[idx]; curr; curr = curr->next) {
        if (curr->support == support && curr->tid_sum == tid_sum) {
            if (is_subset(curr->items, curr->items_len, items, items_len)) return true;
        }
    }
    return false;
}

static void save_generator(TalkyContext *ctx, const uint32_t *items, size_t items_len, uint32_t support, uint32_t tid_sum) {
    uint32_t idx = tid_sum % HASH_SIZE;
    GenNode *node = malloc(sizeof(GenNode));
    node->items = malloc(items_len * sizeof(uint32_t));
    memcpy(node->items, items, items_len * sizeof(uint32_t));
    node->items_len = items_len;
    node->support = support;
    node->tid_sum = tid_sum;
    node->next = ctx->buckets[idx];
    ctx->buckets[idx] = node;
    
    ctx->total_fgs++;
    ctx->total_footprint += items_len;
}

static ITNode* get_next_generator(const ITNode *curr, const ITNode *other, TalkyContext *ctx) {
    // Intersect tids
    uint32_t *new_tids = malloc(curr->tids_len * sizeof(uint32_t));
    size_t new_len = 0;
    uint32_t tid_sum = 0;
    size_t i = 0, j = 0;
    while (i < curr->tids_len && j < other->tids_len) {
        if (curr->tids[i] < other->tids[j]) i++;
        else if (curr->tids[i] > other->tids[j]) j++;
        else {
            tid_sum += curr->tids[i];
            new_tids[new_len++] = curr->tids[i++];
            j++;
        }
    }

    // Test 1: Frequency
    if (new_len < ctx->min_sup) {
        free(new_tids);
        return NULL;
    }

    // Test 2: Generator (support must be strictly less than both parents)
    if (new_len == curr->support || new_len == other->support) {
        free(new_tids);
        return NULL;
    }

    // New items (union)
    // Vertical DFS prefix style: {abc} + {abd} = {abcd}
    size_t new_items_len = curr->items_len + 1;
    uint32_t *new_items = malloc(new_items_len * sizeof(uint32_t));
    memcpy(new_items, curr->items, curr->items_len * sizeof(uint32_t));
    new_items[curr->items_len] = other->items[other->items_len - 1];
    
    // Sort for subset check
    for (size_t x = 0; x < new_items_len; x++) {
        for (size_t y = x + 1; y < new_items_len; y++) {
            if (new_items[x] > new_items[y]) {
                uint32_t tmp = new_items[x];
                new_items[x] = new_items[y];
                new_items[y] = tmp;
            }
        }
    }

    // Test 3: Subsumption check in hash table
    if (test3_subsumption(ctx, new_items, new_items_len, (uint32_t)new_len, tid_sum)) {
        free(new_tids);
        free(new_items);
        return NULL;
    }

    ITNode *node = malloc(sizeof(ITNode));
    node->items = new_items;
    node->items_len = new_items_len;
    node->tids = new_tids;
    node->tids_len = new_len;
    node->support = (uint32_t)new_len;
    node->tid_sum = tid_sum;
    return node;
}

static void talky_extend(ITNode *curr, ITNode *siblings, size_t sib_count, TalkyContext *ctx) {
    if (sib_count == 0) return;

    ITNode *children = malloc(sib_count * sizeof(ITNode));
    size_t child_count = 0;

    // Algorithm 2 line 200: loop over right siblings from left-to-right
    for (size_t i = 0; i < sib_count; i++) {
        ITNode *gen = get_next_generator(curr, &siblings[i], ctx);
        if (gen) {
            children[child_count++] = *gen;
            free(gen);
        }
    }

    // Algorithm 2 line 204: loop over children from right-to-left
    for (int i = (int)child_count - 1; i >= 0; i--) {
        save_generator(ctx, children[i].items, children[i].items_len, children[i].support, children[i].tid_sum);
        // Siblings for the next level are siblings to the RIGHT (children[i+1...])
        // Wait, Algorithm 2 line 200 says siblings of CURR.
        // For children[i], its siblings are those that were created from the same parent.
        // So they are children[i+1...child_count-1]? No, left-to-right order.
        // In reverse pre-order, the siblings of children[i] are children[i+1...].
        talky_extend(&children[i], &children[i + 1], child_count - 1 - i, ctx);
    }

    // Cleanup
    for (size_t i = 0; i < child_count; i++) {
        free(children[i].items);
        free(children[i].tids);
    }
    free(children);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_TALKY_G_Params *p = (DM_TALKY_G_Params *)params;
    double min_sup_param = p ? p->min_support : 0.05;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[Talky-G] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    // 1. Initial 1-itemsets
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    size_t num_frequent = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        // Algorithm 1 line 188: freq AND support < |O|
        if (counts[i] >= min_sup && counts[i] < ds->count) num_frequent++;
    }

    if (num_frequent == 0) {
        printf("[Talky-G] No frequent generators found.\n");
        free(counts);
        return DM_SUCCESS;
    }

    ITNode *root_children = malloc(num_frequent * sizeof(ITNode));
    size_t f_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup && counts[i] < ds->count) {
            root_children[f_idx].items = malloc(sizeof(uint32_t));
            root_children[f_idx].items[0] = i;
            root_children[f_idx].items_len = 1;
            root_children[f_idx].tids = malloc(counts[i] * sizeof(uint32_t));
            root_children[f_idx].tids_len = 0; // Filled below
            root_children[f_idx].support = counts[i];
            root_children[f_idx].tid_sum = 0; // Filled below
            f_idx++;
        }
    }

    // Fill tids
    uint32_t *item_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    memset(item_map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < num_frequent; i++) item_map[root_children[i].items[0]] = (uint32_t)i;

    for (size_t tid = 0; tid < ds->count; tid++) {
        for (size_t j = 0; j < data[tid].count; j++) {
            uint32_t m = item_map[data[tid].items[j]];
            if (m != 0xFFFFFFFF) {
                root_children[m].tids[root_children[m].tids_len++] = (uint32_t)tid;
                root_children[m].tid_sum += (uint32_t)tid;
            }
        }
    }
    free(item_map);

    TalkyContext ctx;
    ctx.buckets = calloc(HASH_SIZE, sizeof(GenNode *));
    ctx.min_sup = min_sup;
    ctx.total_fgs = 0;
    ctx.total_footprint = 0;
    ctx.num_rows = ds->count;

    // Algorithm 1 line 193: loop over children from right-to-left
    for (int i = (int)num_frequent - 1; i >= 0; i--) {
        save_generator(&ctx, root_children[i].items, root_children[i].items_len, root_children[i].support, root_children[i].tid_sum);
        // Siblings for root children are to the RIGHT (root_children[i+1...])
        talky_extend(&root_children[i], &root_children[i + 1], num_frequent - 1 - i, &ctx);
    }

    printf("[Talky-G] Complete. Total frequent generators found: %zu\n", ctx.total_fgs);
    dm_bench_record_results(ctx.total_fgs, ctx.total_footprint);

    // Cleanup
    for (size_t i = 0; i < num_frequent; i++) {
        free(root_children[i].items);
        free(root_children[i].tids);
    }
    free(root_children);
    free(counts);

    for (size_t i = 0; i < HASH_SIZE; i++) {
        GenNode *curr = ctx.buckets[i];
        while (curr) {
            GenNode *next = curr->next;
            free(curr->items);
            free(curr);
            curr = next;
        }
    }
    free(ctx.buckets);

    return DM_SUCCESS;
}

static DM_Algorithm algo_talky = {
    .id = "talky_g",
    .name = "Talky-G Algorithm",
    .description = "Vertical Depth-First Mining of Frequent Generators using Reverse Pre-Order Traversal.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_talky)
