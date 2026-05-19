#include "algorithms/fpmax.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * FP-Max: An Efficient Algorithm for Mining Maximal Frequent Itemsets.
 * Implementation based on Grahne & Zhu (2003/2005).
 * ACM Citation: 1150402.1150473
 */

#define ROOT_ITEM 0xFFFFFFFF

/* --- Data Structures --- */

typedef struct FPNode {
    uint32_t item;
    uint32_t count;
    struct FPNode *parent;
    struct FPNode *children;
    struct FPNode *sibling;
    struct FPNode *next; // Header link
} FPNode;

typedef struct {
    uint32_t item;
    FPNode *head;
    FPNode *tail;
} FPHeader;

typedef struct FPTree {
    FPNode *root;
    FPHeader *header;
    uint32_t header_count;
} FPTree;

typedef struct MFINode {
    uint32_t item;
    struct MFINode *parent;
    struct MFINode *children;
    struct MFINode *sibling;
    struct MFINode *next; // Header link
} MFINode;

typedef struct {
    MFINode *head;
} MFIHeader;

typedef struct MFITree {
    MFINode *root;
    MFIHeader *header; // Indexed by actual item ID
} MFITree;

/* --- Memory Management (Arena/Pool) --- */

typedef struct MemBlock {
    void *ptr;
    size_t size;
    struct MemBlock *next;
} MemBlock;

typedef struct {
    MemBlock *head;
    uint8_t *curr;
    size_t left;
} Arena;

static Arena* arena_create() {
    Arena *a = calloc(1, sizeof(Arena));
    return a;
}

static void* arena_alloc(Arena *a, size_t size) {
    size = (size + 7) & ~7;
    if (a->left < size) {
        size_t block_size = size > 65536 ? size : 65536;
        MemBlock *block = malloc(sizeof(MemBlock));
        block->ptr = malloc(block_size);
        block->size = block_size;
        block->next = a->head;
        a->head = block;
        a->curr = block->ptr;
        a->left = block_size;
    }
    void *ptr = a->curr;
    a->curr += size;
    a->left -= size;
    return ptr;
}

static void arena_destroy(Arena *a) {
    MemBlock *curr = a->head;
    while (curr) {
        MemBlock *next = curr->next;
        free(curr->ptr);
        free(curr);
        curr = next;
    }
    free(a);
}

/* --- Utilities --- */

static uint32_t *global_freq = NULL;

static int cmp_freq_desc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (global_freq[ia] > global_freq[ib]) return -1;
    if (global_freq[ia] < global_freq[ib]) return 1;
    return (ia < ib) ? -1 : 1;
}

/* --- MFI-Tree Logic --- */

static void insert_mfitree(MFITree *mfit, uint32_t *items, uint32_t count, Arena *arena) {
    MFINode *curr = mfit->root;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t item = items[i];
        MFINode *child = curr->children;
        while (child) {
            if (child->item == item) break;
            child = child->sibling;
        }
        if (!child) {
            child = arena_alloc(arena, sizeof(MFINode));
            memset(child, 0, sizeof(MFINode));
            child->item = item;
            child->parent = curr;
            child->sibling = curr->children;
            curr->children = child;
            
            // Header link
            child->next = mfit->header[item].head;
            mfit->header[item].head = child;
        }
        curr = child;
    }
}

/**
 * Check if the set S is a subset of any itemset currently in the MFI-tree.
 * Since items in S are sorted by frequency-descending, and the tree is also 
 * ordered the same way, we can check by starting at all nodes of the least 
 * frequent item in S and traversing upwards.
 */
static bool is_subset_mfi(MFITree *mfit, uint32_t *S, uint32_t count) {
    if (count == 0) return true;
    uint32_t last_item = S[count - 1];
    MFINode *node = mfit->header[last_item].head;
    
    while (node) {
        // Traverse up to see if S is a subset of this path
        MFINode *curr = node;
        int s_idx = (int)count - 1;
        while (curr && curr->item != ROOT_ITEM && s_idx >= 0) {
            if (curr->item == S[s_idx]) {
                s_idx--;
            }
            curr = curr->parent;
        }
        if (s_idx < 0) return true;
        node = node->next;
    }
    return false;
}

/* --- FP-Tree Logic --- */

static void insert_fptree(FPTree *tree, uint32_t *items, uint32_t count, uint32_t weight, uint32_t *item_to_idx, Arena *arena) {
    FPNode *curr = tree->root;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t item = items[i];
        FPNode *child = curr->children;
        while (child) {
            if (child->item == item) break;
            child = child->sibling;
        }
        if (!child) {
            child = arena_alloc(arena, sizeof(FPNode));
            memset(child, 0, sizeof(FPNode));
            child->item = item;
            child->parent = curr;
            child->sibling = curr->children;
            curr->children = child;
            
            // Header link
            uint32_t h_idx = item_to_idx[item];
            if (tree->header[h_idx].tail) {
                tree->header[h_idx].tail->next = child;
                tree->header[h_idx].tail = child;
            } else {
                tree->header[h_idx].head = child;
                tree->header[h_idx].tail = child;
            }
        }
        child->count += weight;
        curr = child;
    }
}

static bool is_single_path(FPTree *tree) {
    FPNode *curr = tree->root;
    while (curr->children) {
        if (curr->children->sibling) return false;
        curr = curr->children;
    }
    return true;
}

/* --- FP-Max Core --- */

typedef struct {
    uint32_t min_sup;
    uint32_t max_id;
    size_t total_mfi;
    size_t total_footprint;
} Context;

static void fpmax_recursive(FPTree *tree, uint32_t *head, uint32_t head_len, MFITree *mfit, Context *ctx, Arena *fp_arena, Arena *mfi_arena) {
    if (is_single_path(tree)) {
        // Collect all items in the path
        uint32_t *S = malloc((head_len + tree->header_count) * sizeof(uint32_t));
        for (uint32_t i = 0; i < head_len; i++) S[i] = head[i];
        
        uint32_t path_len = 0;
        FPNode *curr = tree->root->children;
        while (curr) {
            S[head_len + path_len++] = curr->item;
            curr = curr->children;
        }
        
        uint32_t s_total = head_len + path_len;
        if (s_total > 0) {
            qsort(S, s_total, sizeof(uint32_t), cmp_freq_desc);
            if (!is_subset_mfi(mfit, S, s_total)) {
                insert_mfitree(mfit, S, s_total, mfi_arena);
                ctx->total_mfi++;
                ctx->total_footprint += s_total;
            }
        }
        free(S);
        return;
    }

    // Process header table from bottom to top (least frequent first)
    for (int i = (int)tree->header_count - 1; i >= 0; i--) {
        uint32_t item = tree->header[i].item;
        
        // 1. Calculate Tail(item): frequent items in the conditional database of 'item'
        uint32_t *local_counts = calloc(ctx->max_id + 1, sizeof(uint32_t));
        FPNode *node = tree->header[i].head;
        while (node) {
            uint32_t weight = node->count;
            FPNode *p = node->parent;
            while (p && p->item != ROOT_ITEM) {
                local_counts[p->item] += weight;
                p = p->parent;
            }
            node = node->next;
        }

        uint32_t *tail = malloc((ctx->max_id + 1) * sizeof(uint32_t));
        uint32_t tail_len = 0;
        for (uint32_t j = 0; j <= ctx->max_id; j++) {
            if (local_counts[j] >= ctx->min_sup) {
                tail[tail_len++] = j;
            }
        }

        // 2. MFI-Tree Pruning: If Head U {item} U Tail(item) is a subset of some MFI, prune
        uint32_t s_len = head_len + 1 + tail_len;
        uint32_t *S = malloc(s_len * sizeof(uint32_t));
        for (uint32_t k = 0; k < head_len; k++) S[k] = head[k];
        S[head_len] = item;
        for (uint32_t k = 0; k < tail_len; k++) S[head_len + 1 + k] = tail[k];
        qsort(S, s_len, sizeof(uint32_t), cmp_freq_desc);

        if (!is_subset_mfi(mfit, S, s_len)) {
            if (tail_len == 0) {
                // Maximality guaranteed by is_subset_mfi and order
                insert_mfitree(mfit, S, s_len, mfi_arena);
                ctx->total_mfi++;
                ctx->total_footprint += s_len;
            } else {
                // Build conditional FP-tree
                FPTree cond_tree;
                cond_tree.header_count = tail_len;
                cond_tree.header = malloc(tail_len * sizeof(FPHeader));
                uint32_t *cond_item_to_idx = malloc((ctx->max_id + 1) * sizeof(uint32_t));
                memset(cond_item_to_idx, 0xFF, (ctx->max_id + 1) * sizeof(uint32_t));
                
                // Sort tail to form the header
                qsort(tail, tail_len, sizeof(uint32_t), cmp_freq_desc);
                for (uint32_t k = 0; k < tail_len; k++) {
                    cond_tree.header[k].item = tail[k];
                    cond_tree.header[k].head = NULL;
                    cond_tree.header[k].tail = NULL;
                    cond_item_to_idx[tail[k]] = k;
                }
                
                Arena *cond_arena = arena_create();
                cond_tree.root = arena_alloc(cond_arena, sizeof(FPNode));
                memset(cond_tree.root, 0, sizeof(FPNode));
                cond_tree.root->item = ROOT_ITEM;
                
                // Project transactions
                node = tree->header[i].head;
                while (node) {
                    uint32_t weight = node->count;
                    uint32_t *path = malloc((ctx->max_id + 1) * sizeof(uint32_t));
                    uint32_t path_len = 0;
                    FPNode *p = node->parent;
                    while (p && p->item != ROOT_ITEM) {
                        if (local_counts[p->item] >= ctx->min_sup) {
                            path[path_len++] = p->item;
                        }
                        p = p->parent;
                    }
                    if (path_len > 0) {
                        qsort(path, path_len, sizeof(uint32_t), cmp_freq_desc);
                        insert_fptree(&cond_tree, path, path_len, weight, cond_item_to_idx, cond_arena);
                    }
                    free(path);
                    node = node->next;
                }
                
                uint32_t *new_head = malloc((head_len + 1) * sizeof(uint32_t));
                memcpy(new_head, head, head_len * sizeof(uint32_t));
                new_head[head_len] = item;
                
                fpmax_recursive(&cond_tree, new_head, head_len + 1, mfit, ctx, cond_arena, mfi_arena);
                
                free(new_head);
                free(cond_item_to_idx);
                free(cond_tree.header);
                arena_destroy(cond_arena);
            }
        }
        
        free(S);
        free(tail);
        free(local_counts);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FPMAX_Params *fm_params = (DM_FPMAX_Params *)params;
    double min_sup_param = fm_params ? fm_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[FPmax] Starting. Min Support: %u (%.2f%%)\n", min_sup, (double)min_sup * 100.0 / ds->count);

    // 1. Initial Pass: Count Frequencies
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }
    global_freq = counts;

    // 2. Filter frequent items
    uint32_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }
    if (freq_count == 0) {
        printf("[FPmax] No frequent items found.\n");
        free(counts);
        return DM_SUCCESS;
    }

    uint32_t *L1 = malloc(freq_count * sizeof(uint32_t));
    uint32_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) L1[idx++] = i;
    }
    qsort(L1, freq_count, sizeof(uint32_t), cmp_freq_desc);

    // 3. Build Initial FP-Tree
    Arena *fp_arena = arena_create();
    FPTree tree;
    tree.header_count = freq_count;
    tree.header = malloc(freq_count * sizeof(FPHeader));
    uint32_t *item_to_idx = malloc((ds->max_id + 1) * sizeof(uint32_t));
    memset(item_to_idx, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    
    for (uint32_t i = 0; i < freq_count; i++) {
        tree.header[i].item = L1[i];
        tree.header[i].head = NULL;
        tree.header[i].tail = NULL;
        item_to_idx[L1[i]] = i;
    }
    
    tree.root = arena_alloc(fp_arena, sizeof(FPNode));
    memset(tree.root, 0, sizeof(FPNode));
    tree.root->item = ROOT_ITEM;

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *path = malloc(data[i].count * sizeof(uint32_t));
        uint32_t path_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) path[path_len++] = item;
        }
        if (path_len > 0) {
            qsort(path, path_len, sizeof(uint32_t), cmp_freq_desc);
            insert_fptree(&tree, path, path_len, 1, item_to_idx, fp_arena);
        }
        free(path);
    }

    // 4. Initialize MFI-Tree
    Arena *mfi_arena = arena_create();
    MFITree mfit;
    mfit.root = arena_alloc(mfi_arena, sizeof(MFINode));
    memset(mfit.root, 0, sizeof(MFINode));
    mfit.root->item = ROOT_ITEM;
    mfit.header = calloc(ds->max_id + 1, sizeof(MFIHeader));

    Context ctx = { .min_sup = min_sup, .max_id = ds->max_id, .total_mfi = 0, .total_footprint = 0 };

    // 5. Recursive Mining
    fpmax_recursive(&tree, NULL, 0, &mfit, &ctx, fp_arena, mfi_arena);

    printf("[FPmax] Complete. MFIs found: %zu\n", ctx.total_mfi);
    dm_bench_record_results(ctx.total_mfi, ctx.total_footprint);

    // Cleanup
    free(L1);
    free(item_to_idx);
    free(tree.header);
    free(mfit.header);
    free(counts);
    arena_destroy(fp_arena);
    arena_destroy(mfi_arena);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fpmax",
    .name = "FPmax Algorithm",
    .description = "High Performance Mining of Maximal Frequent Itemsets using MFI-Tree (Grahne & Zhu).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
