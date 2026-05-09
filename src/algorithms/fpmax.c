#include "algorithms/fpmax.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define ROOT_NODE UINT32_MAX

typedef struct FPNode {
    uint32_t item;
    uint32_t count;
    struct FPNode *parent;
    struct FPNode *children;
    struct FPNode *sibling;
    struct FPNode *next;
} FPNode;

typedef struct {
    uint32_t item;
    FPNode *head;
    FPNode *tail;
} FPHeader;

typedef struct FPMemBlock {
    FPNode *nodes;
    size_t used;
    struct FPMemBlock *next;
} FPMemBlock;

typedef struct {
    FPMemBlock *head;
} FPMemPool;

typedef struct {
    FPNode *root;
    FPHeader *header;
    size_t header_count;
    FPMemPool *pool;
} FPTree;

typedef struct MFINode {
    uint32_t item;
    struct MFINode *parent;
    struct MFINode *children;
    struct MFINode *sibling;
    struct MFINode *next;
} MFINode;

typedef struct {
    MFINode *head;
    MFINode *tail;
} MFIHeader;

typedef struct MFIMemBlock {
    MFINode *nodes;
    size_t used;
    struct MFIMemBlock *next;
} MFIMemBlock;

typedef struct {
    MFIMemBlock *head;
} MFIMemPool;

typedef struct {
    MFIHeader *header;
    MFINode *root;
    uint32_t *latest_mfi;
    size_t latest_len;
} MFITree;

typedef struct {
    uint32_t min_sup;
    uint32_t max_id;
    uint32_t *global_counts;
    size_t total_mfi;
    size_t total_footprint;
} FPMaxContext;

static FPNode *alloc_fpnode(FPMemPool *pool) {
    if (!pool->head || pool->head->used >= 4096) {
        FPMemBlock *block = malloc(sizeof(FPMemBlock));
        block->nodes = calloc(4096, sizeof(FPNode));
        block->used = 0;
        block->next = pool->head;
        pool->head = block;
    }
    return &pool->head->nodes[pool->head->used++];
}

static void free_fppool(FPMemPool *pool) {
    FPMemBlock *curr = pool->head;
    while (curr) {
        FPMemBlock *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
    }
}

static MFINode *alloc_mfinode(MFIMemPool *pool) {
    if (!pool->head || pool->head->used >= 4096) {
        MFIMemBlock *block = malloc(sizeof(MFIMemBlock));
        block->nodes = calloc(4096, sizeof(MFINode));
        block->used = 0;
        block->next = pool->head;
        pool->head = block;
    }
    return &pool->head->nodes[pool->head->used++];
}

static void free_mfipool(MFIMemPool *pool) {
    MFIMemBlock *curr = pool->head;
    while (curr) {
        MFIMemBlock *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
    }
}

static uint32_t *g_counts = NULL;
static int cmp_freq_desc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (g_counts[ia] > g_counts[ib]) return -1;
    if (g_counts[ia] < g_counts[ib]) return 1;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static void insert_mfitree(MFITree *tree, uint32_t *items, size_t count, MFIMemPool *pool) {
    MFINode *curr = tree->root;
    for (size_t i = 0; i < count; i++) {
        uint32_t item = items[i];
        MFINode *child = curr->children;
        while (child != NULL) {
            if (child->item == item) break;
            child = child->sibling;
        }
        if (child == NULL) {
            child = alloc_mfinode(pool);
            child->item = item;
            child->parent = curr;
            child->sibling = curr->children;
            curr->children = child;
            
            if (tree->header[item].tail == NULL) {
                tree->header[item].head = child;
                tree->header[item].tail = child;
            } else {
                tree->header[item].tail->next = child;
                tree->header[item].tail = child;
            }
        }
        curr = child;
    }
}

static bool is_subset_mfitree(MFITree *tree, uint32_t *S, size_t count) {
    if (count == 0) return true;
    
    if (tree->latest_mfi) {
        size_t s_idx = 0;
        for (size_t m_idx = 0; m_idx < tree->latest_len && s_idx < count; m_idx++) {
            if (tree->latest_mfi[m_idx] == S[s_idx]) {
                s_idx++;
            }
        }
        if (s_idx == count) return true;
    }
    
    uint32_t in = S[count - 1]; 
    MFINode *node = tree->header[in].head;
    
    while (node != NULL) {
        if (count == 1) return true;
        
        size_t s_idx = count - 1; 
        MFINode *curr = node->parent;
        
        while (s_idx > 0 && curr != tree->root) {
            if (curr->item == S[s_idx - 1]) {
                s_idx--;
            }
            curr = curr->parent;
        }
        
        if (s_idx == 0) return true; 
        
        node = node->next;
    }
    
    return false;
}

static bool fptree_is_single_path(FPTree *tree) {
    if (tree->header_count == 0) return true;
    FPNode *curr = tree->root;
    while (curr) {
        if (curr->children == NULL) return true;
        if (curr->children->sibling != NULL) return false;
        curr = curr->children;
    }
    return true;
}

static FPTree *build_conditional_fptree(FPTree *tree, uint32_t item, uint32_t *counts, FPMaxContext *ctx) {
    FPTree *new_tree = calloc(1, sizeof(FPTree));
    new_tree->pool = calloc(1, sizeof(FPMemPool));
    new_tree->root = alloc_fpnode(new_tree->pool);
    new_tree->root->item = ROOT_NODE;
    
    size_t header_capacity = 0;
    for (uint32_t i = 0; i <= ctx->max_id; i++) {
        if (counts[i] >= ctx->min_sup) header_capacity++;
    }
    new_tree->header = calloc(header_capacity, sizeof(FPHeader));
    new_tree->header_count = header_capacity;
    
    uint32_t *freq_items = malloc(header_capacity * sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t i = 0; i <= ctx->max_id; i++) {
        if (counts[i] >= ctx->min_sup) freq_items[idx++] = i;
    }
    qsort(freq_items, header_capacity, sizeof(uint32_t), cmp_freq_desc);
    
    for (size_t i = 0; i < header_capacity; i++) {
        new_tree->header[i].item = freq_items[i];
    }
    
    uint32_t *item_to_idx = malloc((ctx->max_id + 1) * sizeof(uint32_t));
    memset(item_to_idx, 0xFF, (ctx->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < header_capacity; i++) {
        item_to_idx[freq_items[i]] = i;
    }
    
    FPNode *target_head = NULL;
    for (size_t i = 0; i < tree->header_count; i++) {
        if (tree->header[i].item == item) {
            target_head = tree->header[i].head;
            break;
        }
    }
    
    FPNode *node = target_head;
    while (node) {
        uint32_t weight = node->count;
        
        uint32_t *path = malloc(ctx->max_id * sizeof(uint32_t));
        size_t path_len = 0;
        FPNode *curr = node->parent;
        while (curr && curr->item != ROOT_NODE) {
            if (counts[curr->item] >= ctx->min_sup) {
                path[path_len++] = curr->item;
            }
            curr = curr->parent;
        }
        
        if (path_len > 0) {
            qsort(path, path_len, sizeof(uint32_t), cmp_freq_desc);
            
            FPNode *insert_curr = new_tree->root;
            for (size_t k = 0; k < path_len; k++) {
                uint32_t p_item = path[k];
                FPNode *child = insert_curr->children;
                while (child) {
                    if (child->item == p_item) break;
                    child = child->sibling;
                }
                if (!child) {
                    child = alloc_fpnode(new_tree->pool);
                    child->item = p_item;
                    child->parent = insert_curr;
                    child->sibling = insert_curr->children;
                    insert_curr->children = child;
                    
                    uint32_t h_idx = item_to_idx[p_item];
                    if (new_tree->header[h_idx].tail == NULL) {
                        new_tree->header[h_idx].head = child;
                        new_tree->header[h_idx].tail = child;
                    } else {
                        new_tree->header[h_idx].tail->next = child;
                        new_tree->header[h_idx].tail = child;
                    }
                }
                child->count += weight;
                insert_curr = child;
            }
        }
        
        free(path);
        node = node->next;
    }
    
    free(freq_items);
    free(item_to_idx);
    
    return new_tree;
}

static void fpmax(FPTree *tree, uint32_t *head, size_t head_len, MFITree *mfit, FPMaxContext *ctx, MFIMemPool *mpool) {
    if (fptree_is_single_path(tree)) {
        uint32_t *P = malloc((tree->header_count + 1) * sizeof(uint32_t));
        size_t P_len = 0;
        FPNode *curr = tree->root->children;
        while (curr) {
            P[P_len++] = curr->item;
            curr = curr->children;
        }
        
        uint32_t *new_MFI = malloc((head_len + P_len) * sizeof(uint32_t));
        for (size_t k = 0; k < head_len; k++) new_MFI[k] = head[k];
        for (size_t k = 0; k < P_len; k++) new_MFI[head_len + k] = P[k];
        
        if (head_len + P_len > 0) {
            qsort(new_MFI, head_len + P_len, sizeof(uint32_t), cmp_freq_desc);
            
            insert_mfitree(mfit, new_MFI, head_len + P_len, mpool);
            
            ctx->total_mfi++;
            ctx->total_footprint += head_len + P_len;
            
            if (mfit->latest_mfi) free(mfit->latest_mfi);
            mfit->latest_mfi = new_MFI;
            mfit->latest_len = head_len + P_len;
        } else {
            free(new_MFI);
        }
        free(P);
        return;
    }
    
    for (int i = tree->header_count - 1; i >= 0; i--) {
        uint32_t item = tree->header[i].item;
        
        uint32_t *counts = calloc(ctx->max_id + 1, sizeof(uint32_t));
        FPNode *node = tree->header[i].head;
        while (node) {
            uint32_t weight = node->count;
            FPNode *curr = node->parent;
            while (curr && curr->item != ROOT_NODE) {
                counts[curr->item] += weight;
                curr = curr->parent;
            }
            node = node->next;
        }
        
        uint32_t *Tail = malloc((ctx->max_id + 1) * sizeof(uint32_t));
        size_t tail_len = 0;
        for (uint32_t j = 0; j <= ctx->max_id; j++) {
            if (counts[j] >= ctx->min_sup) {
                Tail[tail_len++] = j;
            }
        }
        
        uint32_t *S = malloc((head_len + 1 + tail_len) * sizeof(uint32_t));
        size_t S_len = 0;
        for (size_t k = 0; k < head_len; k++) S[S_len++] = head[k];
        S[S_len++] = item;
        for (size_t k = 0; k < tail_len; k++) S[S_len++] = Tail[k];
        
        qsort(S, S_len, sizeof(uint32_t), cmp_freq_desc);
        
        if (!is_subset_mfitree(mfit, S, S_len)) {
            if (tail_len == 0) {
                uint32_t *new_MFI = malloc(S_len * sizeof(uint32_t));
                memcpy(new_MFI, S, S_len * sizeof(uint32_t));
                insert_mfitree(mfit, new_MFI, S_len, mpool);
                ctx->total_mfi++;
                ctx->total_footprint += S_len;
                if (mfit->latest_mfi) free(mfit->latest_mfi);
                mfit->latest_mfi = new_MFI;
                mfit->latest_len = S_len;
            } else {
                FPTree *T_item = build_conditional_fptree(tree, item, counts, ctx);
                
                uint32_t *new_head = malloc((head_len + 1) * sizeof(uint32_t));
                memcpy(new_head, head, head_len * sizeof(uint32_t));
                new_head[head_len] = item;
                
                fpmax(T_item, new_head, head_len + 1, mfit, ctx, mpool);
                
                free_fppool(T_item->pool);
                free(T_item->header);
                free(T_item->pool);
                free(T_item);
                free(new_head);
            }
        }
        
        free(S);
        free(Tail);
        free(counts);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FPMAX_Params *fm_params = (DM_FPMAX_Params *)params;
    double min_sup_param = fm_params ? fm_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[FPmax] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }
    
    g_counts = counts;

    uint32_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        printf("[FPmax] Complete. Total maximal frequent itemsets found: 0\n");
        free(counts);
        return DM_SUCCESS;
    }

    uint32_t *L1 = malloc(freq_count * sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) L1[idx++] = i;
    }
    qsort(L1, freq_count, sizeof(uint32_t), cmp_freq_desc);

    FPMaxContext ctx = {0};
    ctx.min_sup = min_sup;
    ctx.max_id = ds->max_id;
    ctx.global_counts = counts;

    FPTree *initial_tree = calloc(1, sizeof(FPTree));
    initial_tree->pool = calloc(1, sizeof(FPMemPool));
    initial_tree->root = alloc_fpnode(initial_tree->pool);
    initial_tree->root->item = ROOT_NODE;
    
    initial_tree->header_count = freq_count;
    initial_tree->header = calloc(freq_count, sizeof(FPHeader));
    for (size_t i = 0; i < freq_count; i++) {
        initial_tree->header[i].item = L1[i];
    }
    
    uint32_t *item_to_idx = malloc((ctx.max_id + 1) * sizeof(uint32_t));
    memset(item_to_idx, 0xFF, (ctx.max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < freq_count; i++) {
        item_to_idx[L1[i]] = i;
    }
    
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *path = malloc(data[i].count * sizeof(uint32_t));
        size_t path_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) {
                path[path_len++] = item;
            }
        }
        
        if (path_len > 0) {
            qsort(path, path_len, sizeof(uint32_t), cmp_freq_desc);
            
            FPNode *insert_curr = initial_tree->root;
            for (size_t k = 0; k < path_len; k++) {
                uint32_t p_item = path[k];
                FPNode *child = insert_curr->children;
                while (child) {
                    if (child->item == p_item) break;
                    child = child->sibling;
                }
                if (!child) {
                    child = alloc_fpnode(initial_tree->pool);
                    child->item = p_item;
                    child->parent = insert_curr;
                    child->sibling = insert_curr->children;
                    insert_curr->children = child;
                    
                    uint32_t h_idx = item_to_idx[p_item];
                    if (initial_tree->header[h_idx].tail == NULL) {
                        initial_tree->header[h_idx].head = child;
                        initial_tree->header[h_idx].tail = child;
                    } else {
                        initial_tree->header[h_idx].tail->next = child;
                        initial_tree->header[h_idx].tail = child;
                    }
                }
                child->count++;
                insert_curr = child;
            }
        }
        free(path);
    }

    MFITree mfit = {0};
    mfit.header = calloc(ctx.max_id + 1, sizeof(MFIHeader));
    MFIMemPool mpool = {0};
    mfit.root = alloc_mfinode(&mpool);
    mfit.root->item = ROOT_NODE;

    fpmax(initial_tree, NULL, 0, &mfit, &ctx, &mpool);

    printf("[FPmax] Complete. Total maximal frequent itemsets found: %zu\n", ctx.total_mfi);
    dm_bench_record_results(ctx.total_mfi, ctx.total_footprint);

    if (mfit.latest_mfi) free(mfit.latest_mfi);
    free_mfipool(&mpool);
    free(mfit.header);

    free_fppool(initial_tree->pool);
    free(initial_tree->header);
    free(initial_tree->pool);
    free(initial_tree);
    
    free(item_to_idx);
    free(L1);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fpmax",
    .name = "FPmax Algorithm",
    .description = "High Performance Mining of Maximal Frequent Itemsets using MFI-Tree.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
