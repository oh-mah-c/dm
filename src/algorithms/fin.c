#include "algorithms/fin.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * FIN: Fast Information-lossless Itemset Non-redundant Mining.
 * Based on: "FIN: A Fast Algorithm for Mining Frequent Itemsets using Nodesets" 
 * by Zhi-Hong Deng and Sheng-Long Lv (2014).
 */

/* --- Data Structures --- */

typedef struct PPCNode {
    uint32_t item;
    uint32_t count;
    uint32_t pre;
    uint32_t post;
    struct PPCNode *parent;
    struct PPCNode *children;
    struct PPCNode *sibling;
} PPCNode;

typedef struct {
    uint32_t pre;
    uint32_t post;
    uint32_t count;
} NodeInfo;

typedef struct {
    uint32_t item;
    NodeInfo *nodes;
    uint32_t node_count;
    uint32_t support;
} Nodeset;

typedef struct {
    uint32_t min_sup;
    uint32_t max_id;
    size_t total_fi;
    size_t total_footprint;
} Context;

/* --- Memory Management --- */

typedef struct MemBlock {
    PPCNode *nodes;
    size_t used;
    struct MemBlock *next;
} MemBlock;

typedef struct {
    MemBlock *head;
} Pool;

static PPCNode* alloc_node(Pool *pool) {
    if (!pool->head || pool->head->used >= 4096) {
        MemBlock *block = malloc(sizeof(MemBlock));
        block->nodes = calloc(4096, sizeof(PPCNode));
        block->used = 0;
        block->next = pool->head;
        pool->head = block;
    }
    return &pool->head->nodes[pool->head->used++];
}

static void free_pool(Pool *pool) {
    MemBlock *curr = pool->head;
    while (curr) {
        MemBlock *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
    }
}

/* --- PPC-Tree Logic --- */

static uint32_t g_pre = 0;
static uint32_t g_post = 0;

static void traverse_ppc(PPCNode *node) {
    if (!node) return;
    node->pre = ++g_pre;
    PPCNode *child = node->children;
    while (child) {
        traverse_ppc(child);
        child = child->sibling;
    }
    node->post = ++g_post;
}

/* --- Nodeset Intersection --- */

/**
 * Intersect Nodeset(X) and Nodeset(Y) to get Nodeset(XY).
 * Logic: For each node ny in NS(Y), if there exists nx in NS(X) such that nx is an ancestor of ny,
 * then ny is added to NS(XY).
 * nx is ancestor of ny iff nx.pre < ny.pre and nx.post > ny.post.
 */
static Nodeset intersect(Nodeset *X, Nodeset *Y) {
    Nodeset res;
    res.item = Y->item;
    res.nodes = malloc(Y->node_count * sizeof(NodeInfo));
    res.node_count = 0;
    res.support = 0;

    uint32_t i = 0, j = 0;
    while (i < X->node_count && j < Y->node_count) {
        NodeInfo *nx = &X->nodes[i];
        NodeInfo *ny = &Y->nodes[j];

        if (nx->pre < ny->pre) {
            if (nx->post > ny->post) {
                // nx is ancestor of ny
                res.nodes[res.node_count++] = *ny;
                res.support += ny->count;
                j++; // Move to next node in Y
            } else {
                // nx is not ancestor, and cannot be ancestor of any subsequent ny (since Y is sorted by pre)
                // Actually, wait. Y is sorted by pre.
                // If nx.post < ny.post, then nx cannot be ancestor of ny.
                // Could nx be ancestor of Y[j+1]? No, because Y[j+1].pre > ny.pre.
                // Wait, if nx.pre < ny.pre and nx.post < ny.post, it's possible nx is ancestor of nothing?
                // We should increment i.
                i++;
            }
        } else {
            // ny.pre <= nx.pre. ny cannot have nx as ancestor.
            j++;
        }
    }
    return res;
}

/* --- Mining --- */

static void fin_recursive(Nodeset *prefix_ns, uint32_t prefix_len, Nodeset *candidates, uint32_t cand_count, Context *ctx) {
    for (uint32_t i = 0; i < cand_count; i++) {
        Nodeset new_ns = intersect(prefix_ns, &candidates[i]);
        if (new_ns.support >= ctx->min_sup) {
            ctx->total_fi++;
            ctx->total_footprint += (prefix_len + 1);
            
            if (i + 1 < cand_count) {
                fin_recursive(&new_ns, prefix_len + 1, candidates + i + 1, cand_count - i - 1, ctx);
            }
        }
        free(new_ns.nodes);
    }
}

static uint32_t *global_counts = NULL;
static int cmp_freq_desc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (global_counts[ia] > global_counts[ib]) return -1;
    if (global_counts[ia] < global_counts[ib]) return 1;
    return (ia < ib) ? -1 : 1;
}

static void collect_nodes(PPCNode *node, Nodeset *ns) {
    if (!node) return;
    if (node->item != 0xFFFFFFFF) {
        uint32_t item = node->item;
        ns[item].node_count++;
    }
    PPCNode *child = node->children;
    while (child) {
        collect_nodes(child, ns);
        child = child->sibling;
    }
}

static void fill_nodes(PPCNode *node, Nodeset *ns) {
    if (!node) return;
    if (node->item != 0xFFFFFFFF) {
        uint32_t item = node->item;
        NodeInfo *ni = &ns[item].nodes[ns[item].node_count++];
        ni->pre = node->pre;
        ni->post = node->post;
        ni->count = node->count;
        ns[item].support += node->count;
        ns[item].item = item;
    }
    PPCNode *child = node->children;
    while (child) {
        fill_nodes(child, ns);
        child = child->sibling;
    }
}

static int cmp_node_info(const void *a, const void *b) {
    uint32_t pa = ((const NodeInfo *)a)->pre;
    uint32_t pb = ((const NodeInfo *)b)->pre;
    return (pa < pb) ? -1 : 1;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FIN_Params *f_params = (DM_FIN_Params *)params;
    double min_sup_param = f_params ? f_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[FIN] Starting. Min Support: %u\n", min_sup);

    // 1. Initial Pass: Count Frequencies
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }
    global_counts = counts;

    uint32_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }
    if (freq_count == 0) {
        free(counts);
        return DM_SUCCESS;
    }

    uint32_t *L1 = malloc(freq_count * sizeof(uint32_t));
    uint32_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) L1[idx++] = i;
    }
    qsort(L1, freq_count, sizeof(uint32_t), cmp_freq_desc);

    // 2. Build PPC-Tree
    Pool pool = {0};
    PPCNode *root = alloc_node(&pool);
    memset(root, 0, sizeof(PPCNode));
    root->item = 0xFFFFFFFF;

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *path = malloc(data[i].count * sizeof(uint32_t));
        uint32_t path_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) path[path_len++] = item;
        }
        if (path_len > 0) {
            qsort(path, path_len, sizeof(uint32_t), cmp_freq_desc);
            PPCNode *curr = root;
            for (uint32_t k = 0; k < path_len; k++) {
                uint32_t item = path[k];
                PPCNode *child = curr->children;
                while (child) {
                    if (child->item == item) break;
                    child = child->sibling;
                }
                if (!child) {
                    child = alloc_node(&pool);
                    memset(child, 0, sizeof(PPCNode));
                    child->item = item;
                    child->parent = curr;
                    child->sibling = curr->children;
                    curr->children = child;
                }
                child->count++;
                curr = child;
            }
        }
        free(path);
    }

    // 3. Assign Pre/Post Codes
    g_pre = 0; g_post = 0;
    traverse_ppc(root);

    // 4. Collect Initial Nodesets
    Nodeset *item_nodesets = calloc(ds->max_id + 1, sizeof(Nodeset));
    // We need to traverse the whole tree or use a header table. 
    // Let's use a queue or recursion to collect.
    
    collect_nodes(root, item_nodesets);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_nodesets[i].node_count > 0) {
            item_nodesets[i].nodes = malloc(item_nodesets[i].node_count * sizeof(NodeInfo));
            item_nodesets[i].node_count = 0; // reset to use as index
        }
    }
    fill_nodes(root, item_nodesets);

    // Nodesets must be sorted by pre-order for the intersection logic to work efficiently.
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_nodesets[i].node_count > 0) {
            qsort(item_nodesets[i].nodes, item_nodesets[i].node_count, sizeof(NodeInfo), cmp_node_info);
        }
    }

    // 5. Mining
    Context ctx = { .min_sup = min_sup, .max_id = ds->max_id, .total_fi = 0, .total_footprint = 0 };
    Nodeset *candidates = malloc(freq_count * sizeof(Nodeset));
    for (uint32_t i = 0; i < freq_count; i++) {
        candidates[i] = item_nodesets[L1[i]];
        ctx.total_fi++;
        ctx.total_footprint += 1;
    }
    
    for (uint32_t i = 0; i < freq_count; i++) {
        if (i + 1 < freq_count) {
            fin_recursive(&candidates[i], 1, candidates + i + 1, freq_count - i - 1, &ctx);
        }
    }

    printf("[FIN] Complete. FIs found: %zu\n", ctx.total_fi);
    dm_bench_record_results(ctx.total_fi, ctx.total_footprint);

    // Cleanup
    for (uint32_t i = 0; i <= ds->max_id; i++) if (item_nodesets[i].nodes) free(item_nodesets[i].nodes);
    free(item_nodesets);
    free(candidates);
    free(L1);
    free(counts);
    free_pool(&pool);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fin",
    .name = "FIN Algorithm",
    .description = "Fast Mining of Frequent Itemsets using Nodesets (Deng & Lv).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
