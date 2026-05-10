#include "algorithms/negfin.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * negFIN: An efficient algorithm for fast mining frequent itemsets.
 * Reference: Nader Aryabarzan et al., "negFIN: An efficient algorithm for fast mining frequent itemsets", 
 * Expert Systems With Applications 105 (2018) 129-143.
 */

/* --- Bitmap Utilities --- */

typedef struct {
    uint64_t *data;
} Bitmap;

static inline bool bitmap_test(const Bitmap b, uint32_t index) {
    return (b.data[index >> 6] >> (index & 0x3F)) & 1;
}

static inline void bitmap_set(Bitmap b, uint32_t index) {
    b.data[index >> 6] |= (1ULL << (index & 0x3F));
}

static inline void bitmap_copy(Bitmap dst, const Bitmap src, uint32_t words) {
    memcpy(dst.data, src.data, words * sizeof(uint64_t));
}

/* --- Data Structures --- */

typedef struct BMC_Node {
    uint32_t item;
    uint32_t count;
    Bitmap bitmap_code;
    struct BMC_Node *parent;
    struct BMC_Node *children;
    struct BMC_Node *sibling;
} BMC_Node;

typedef struct {
    Bitmap bitmap_code;
    uint32_t count;
} N_info;

typedef struct {
    N_info *infos;
    uint32_t len;
    uint32_t support;
} Nodeset;

typedef struct {
    uint32_t min_sup;
    size_t total_fi;
    size_t total_footprint;
    uint32_t nf; 
    uint32_t bitmap_words;
    uint32_t *L1;
    uint32_t *rank_map;
} Context;

typedef struct {
    uint32_t item;
    Nodeset ns;
    uint32_t support;
} FI_Node;

/* --- Memory Management --- */

typedef struct MemBlock {
    BMC_Node *nodes;
    size_t used;
    struct MemBlock *next;
} MemBlock;

typedef struct BitmapBlock {
    uint64_t *data;
    size_t used;
    struct BitmapBlock *next;
} BitmapBlock;

typedef struct {
    MemBlock *node_head;
    BitmapBlock *bitmap_head;
    uint32_t bitmap_words;
} Pool;

static BMC_Node* alloc_bmc_node(Pool *pool) {
    if (!pool->node_head || pool->node_head->used >= 4096) {
        MemBlock *block = malloc(sizeof(MemBlock));
        block->nodes = calloc(4096, sizeof(BMC_Node));
        block->used = 0;
        block->next = pool->node_head;
        pool->node_head = block;
    }
    return &pool->node_head->nodes[pool->node_head->used++];
}

static uint64_t* alloc_bitmap(Pool *pool) {
    size_t words_per_block = 65536;
    if (!pool->bitmap_head || pool->bitmap_head->used + pool->bitmap_words > words_per_block) {
        BitmapBlock *block = malloc(sizeof(BitmapBlock));
        block->data = calloc(words_per_block, sizeof(uint64_t));
        block->used = 0;
        block->next = pool->bitmap_head;
        pool->bitmap_head = block;
    }
    uint64_t *ptr = &pool->bitmap_head->data[pool->bitmap_head->used];
    pool->bitmap_head->used += pool->bitmap_words;
    return ptr;
}

static void free_pool(Pool *pool) {
    MemBlock *curr_n = pool->node_head;
    while (curr_n) {
        MemBlock *next = curr_n->next;
        free(curr_n->nodes);
        free(curr_n);
        curr_n = next;
    }
    BitmapBlock *curr_b = pool->bitmap_head;
    while (curr_b) {
        BitmapBlock *next = curr_b->next;
        free(curr_b->data);
        free(curr_b);
        curr_b = next;
    }
}

/* --- Global State for Sort --- */

static uint32_t *g_counts = NULL;
static int cmp_support_asc(const void *a, const void *b) {
    uint32_t ia = *(uint32_t*)a, ib = *(uint32_t*)b;
    if (g_counts[ia] < g_counts[ib]) return -1;
    if (g_counts[ia] > g_counts[ib]) return 1;
    return (ia < ib) ? -1 : 1;
}

static uint32_t *g_rank_map = NULL;
static int cmp_rank_rev(const void *a, const void *b) {
    return (int)g_rank_map[*(uint32_t*)b] - (int)g_rank_map[*(uint32_t*)a];
}

/* --- Algorithm 1: constructing_BMC_tree --- */

static BMC_Node* constructing_BMC_tree(DM_Dataset *ds, uint32_t min_sup, Context *ctx, Pool *pool) {
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple*)ds->payload;
    for (size_t i = 0; i < ds->count; i++)
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;

    uint32_t freq_cnt = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_cnt++;
    if (freq_cnt == 0) { free(counts); return NULL; }

    uint32_t *L1 = malloc(freq_cnt * sizeof(uint32_t));
    uint32_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) L1[idx++] = i;
    
    g_counts = counts;
    qsort(L1, freq_cnt, sizeof(uint32_t), cmp_support_asc);

    ctx->nf = freq_cnt;
    ctx->bitmap_words = (freq_cnt + 63) / 64;
    ctx->L1 = L1;
    ctx->rank_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) ctx->rank_map[i] = 0xFFFFFFFF;
    for (uint32_t i = 0; i < freq_cnt; i++) ctx->rank_map[L1[i]] = i;

    pool->bitmap_words = ctx->bitmap_words;
    g_rank_map = ctx->rank_map;

    BMC_Node *root = alloc_bmc_node(pool);
    root->item = 0xFFFFFFFF;
    root->count = 0;
    root->bitmap_code.data = alloc_bitmap(pool);

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *T = malloc(data[i].count * sizeof(uint32_t));
        uint32_t t_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (ctx->rank_map[data[i].items[j]] != 0xFFFFFFFF) T[t_len++] = data[i].items[j];
        }
        if (t_len > 0) {
            qsort(T, t_len, sizeof(uint32_t), cmp_rank_rev);
            BMC_Node *curr = root;
            for (uint32_t j = 0; j < t_len; j++) {
                uint32_t item = T[j];
                BMC_Node *child = curr->children, *prev = NULL;
                while (child && child->item != item) { prev = child; child = child->sibling; }
                if (!child) {
                    child = alloc_bmc_node(pool);
                    child->item = item;
                    child->parent = curr;
                    child->bitmap_code.data = alloc_bitmap(pool);
                    bitmap_copy(child->bitmap_code, curr->bitmap_code, ctx->bitmap_words);
                    bitmap_set(child->bitmap_code, ctx->rank_map[item]);
                    if (prev) prev->sibling = child; else curr->children = child;
                }
                child->count++;
                curr = child;
            }
        }
        free(T);
    }
    free(counts);
    return root;
}

/* --- NegNodeset Utilities --- */

static void collect_neg_nodeset(Nodeset *R, const Nodeset *P, uint32_t item_index, bool level1) {
    R->infos = malloc(P->len * sizeof(N_info));
    R->len = 0;
    R->support = 0;
    for (uint32_t i = 0; i < P->len; i++) {
        bool condition;
        if (level1) {
            condition = !bitmap_test(P->infos[i].bitmap_code, item_index);
        } else {
            condition = bitmap_test(P->infos[i].bitmap_code, item_index);
        }
        if (condition) {
            R->infos[R->len++] = P->infos[i];
            R->support += P->infos[i].count;
        }
    }
}

/* --- Algorithm 4: constructing_frequent_itemset_tree --- */

static void constructing_frequent_itemset_tree(FI_Node *N, FI_Node *siblings, uint32_t sibling_count, size_t FIS_parent_count, Context *ctx) {
    uint32_t *equiv = malloc(ctx->nf * sizeof(uint32_t));
    uint32_t equiv_count = 0;
    uint32_t n_idx = ctx->rank_map[N->item];
    
    FI_Node *children = malloc(ctx->nf * sizeof(FI_Node));
    uint32_t child_count = 0;

    for (uint32_t i = n_idx + 1; i < ctx->nf; i++) {
        uint32_t item_i = ctx->L1[i];
        Nodeset R_neg;

        if (siblings == NULL) { 
            collect_neg_nodeset(&R_neg, &N->ns, i, true);
        } else {
            FI_Node *Q = NULL;
            for (uint32_t k = 0; k < sibling_count; k++) {
                if (siblings[k].item == item_i) {
                    Q = &siblings[k];
                    break;
                }
            }
            if (Q) {
                collect_neg_nodeset(&R_neg, &Q->ns, n_idx, false);
            } else {
                continue;
            }
        }

        uint32_t R_support = N->support - R_neg.support;
        
        if (R_support == N->support) {
            equiv[equiv_count++] = item_i;
            free(R_neg.infos);
        } else if (R_support >= ctx->min_sup) {
            children[child_count].item = item_i;
            children[child_count].ns = R_neg;
            children[child_count].support = R_support;
            child_count++;
        } else {
            free(R_neg.infos);
        }
    }

    // Line 39-45: Identify all frequent itemsets in N
    size_t num_equiv_subsets = (size_t)1 << equiv_count;
    size_t base_count = (siblings == NULL) ? 1 : FIS_parent_count;
    size_t FIS_N_count = num_equiv_subsets * base_count;
    
    if (siblings == NULL) {
        // P is a 1-itemset. P itself is already in F (ctx->nf). 
        // We only add new extensions.
        ctx->total_fi += (FIS_N_count - 1);
    } else {
        ctx->total_fi += FIS_N_count;
    }

    for (uint32_t i = 0; i < child_count; i++) {
        constructing_frequent_itemset_tree(&children[i], children, child_count, FIS_N_count, ctx);
    }

    for (uint32_t i = 0; i < child_count; i++) free(children[i].ns.infos);
    free(children);
    free(equiv);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_NEGFIN_Params *p = (DM_NEGFIN_Params*)params;
    double ms_val = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_val < 1.0) ? (uint32_t)ceil(ms_val * ds->count) : (uint32_t)ms_val;
    if (min_sup == 0) min_sup = 1;

    printf("[negFIN] Starting. Min Support: %u\n", min_sup);

    Pool pool = {0};
    Context ctx = { .min_sup = min_sup, .total_fi = 0, .total_footprint = 0 };

    BMC_Node *root = constructing_BMC_tree(ds, min_sup, &ctx, &pool);
    if (!root) { free_pool(&pool); return DM_SUCCESS; }
    
    ctx.total_fi = ctx.nf;

    Nodeset *f1_ns = calloc(ctx.nf, sizeof(Nodeset));
    for (uint32_t i = 0; i < ctx.nf; i++) {
        f1_ns[i].infos = malloc(ds->count * sizeof(N_info));
        f1_ns[i].len = 0;
        f1_ns[i].support = 0;
    }

    void collect_f1_ns_recursive(BMC_Node *n, Nodeset *ns_array, uint32_t *rm) {
        if (!n) return;
        if (n->item != 0xFFFFFFFF) {
            uint32_t r = rm[n->item];
            ns_array[r].infos[ns_array[r].len++] = (N_info){n->bitmap_code, n->count};
            ns_array[r].support += n->count;
        }
        BMC_Node *c = n->children;
        while (c) { collect_f1_ns_recursive(c, ns_array, rm); c = c->sibling; }
    }
    collect_f1_ns_recursive(root, f1_ns, ctx.rank_map);

    FI_Node *f1_nodes = malloc(ctx.nf * sizeof(FI_Node));
    for (uint32_t i = 0; i < ctx.nf; i++) {
        f1_nodes[i].item = ctx.L1[i];
        f1_nodes[i].ns = f1_ns[i];
        f1_nodes[i].support = f1_ns[i].support;
    }

    for (uint32_t i = 0; i < ctx.nf; i++) {
        constructing_frequent_itemset_tree(&f1_nodes[i], NULL, 0, 0, &ctx);
    }

    printf("[negFIN] Complete. FIs found: %zu\n", ctx.total_fi);
    dm_bench_record_results(ctx.total_fi, ctx.total_footprint);

    for (uint32_t i = 0; i < ctx.nf; i++) free(f1_ns[i].infos);
    free(f1_ns); free(f1_nodes); free(ctx.L1); free(ctx.rank_map); free_pool(&pool);
    return DM_SUCCESS;
}

static DM_Algorithm algo_negfin = {
    .id = "negfin", .name = "negFIN Algorithm",
    .description = "An efficient algorithm for fast mining frequent itemsets using NegNodesets.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo_negfin)
