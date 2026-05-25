#include "algorithms/finplus.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * FIN+ (dFIN): Fast Mining of Frequent Itemsets using DiffNodesets.
 * Reference: Zhi-Hong Deng, "DiffNodesets: An Efficient Structure for Fast Mining Frequent Itemsets", 2016.
 * DOI: 10.1016/j.asoc.2016.03.023 (arXiv:1507.01345)
 */

#define ROOT_ITEM 0xFFFFFFFF

/* --- Data Structures --- */

typedef struct PPC_Node {
    uint32_t item;
    uint32_t count;
    uint32_t pre;
    uint32_t post;
    struct PPC_Node *parent;
    struct PPC_Node *children;
    struct PPC_Node *sibling;
} PPC_Node;

typedef struct {
    uint32_t pre;
    uint32_t post;
    uint32_t count;
} PP_Code;

typedef struct {
    PP_Code *codes;
    uint32_t len;
    uint32_t support;
} Nodeset;

typedef struct {
    PP_Code *codes;
    uint32_t len;
    uint32_t sum_count; // Sum of counts in the DiffNodeset
} DiffNodeset;

typedef struct {
    uint32_t *items;
    uint32_t len;
    uint32_t support;
    DiffNodeset dn;
} Itemset;

typedef struct {
    uint32_t min_sup;
    size_t total_fi;
    size_t total_footprint;
} Context;

/* --- Memory Pool for PPC_Node --- */

typedef struct MemBlock {
    PPC_Node *nodes;
    size_t used;
    struct MemBlock *next;
} MemBlock;

typedef struct {
    MemBlock *head;
} Pool;

static PPC_Node* alloc_ppc_node(Pool *pool) {
    if (!pool->head || pool->head->used >= 16384) {
        MemBlock *block = malloc(sizeof(MemBlock));
        block->nodes = calloc(16384, sizeof(PPC_Node));
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

/* --- Tree Construction & Traversal --- */

static uint32_t g_pre = 0, g_post = 0;
static void traverse_ppc(PPC_Node *n) {
    if (!n) return;
    n->pre = ++g_pre;
    PPC_Node *c = n->children;
    while (c) {
        traverse_ppc(c);
        c = c->sibling;
    }
    n->post = ++g_post;
}

/* --- Core Algorithms (Paper Procedures) --- */

/**
 * Build_2-itemset_DN (Page 8)
 * Generates DNxy = NSx \ Anc(NSy)
 */
static DiffNodeset build_2_itemset_dn(Nodeset *NSx, Nodeset *NSy) {
    DiffNodeset res;
    res.codes = malloc(NSx->len * sizeof(PP_Code));
    res.len = 0;
    res.sum_count = 0;

    uint32_t k = 0, j = 0;
    while (k < NSx->len && j < NSy->len) {
        // Nx[k] vs Ny[j]
        if (NSx->codes[k].post > NSy->codes[j].post) {
            // Case 1 (Line 5)
            j++;
        } else {
            // Else (Line 7)
            if (NSx->codes[k].post < NSy->codes[j].post && NSx->codes[k].pre > NSy->codes[j].pre) {
                // Case 2 (Line 8): ny is ancestor of nx
                k++;
            } else {
                // Else (Line 10): nx is desirable
                res.codes[res.len] = NSx->codes[k];
                res.sum_count += NSx->codes[k].count;
                res.len++;
                k++;
            }
        }
    }
    // Append remaining (Line 16-21)
    while (k < NSx->len) {
        res.codes[res.len] = NSx->codes[k];
        res.sum_count += NSx->codes[k].count;
        res.len++;
        k++;
    }
    return res;
}

/**
 * Set Difference for k-itemsets (Theorem 4, Page 10)
 * DNP = DN2 / DN1
 * Where DN1, DN2 are DiffNodesets from the previous level.
 * Since they are sorted by pre-order (which corresponds to post-order logic in FIN+),
 * we can use a similar merge-style difference.
 */
static DiffNodeset set_difference_dn(DiffNodeset *DN2, DiffNodeset *DN1) {
    DiffNodeset res;
    res.codes = malloc(DN2->len * sizeof(PP_Code));
    res.len = 0;
    res.sum_count = 0;

    uint32_t k = 0, j = 0;
    while (k < DN2->len && j < DN1->len) {
        if (DN2->codes[k].pre < DN1->codes[j].pre) {
            // DN2 has it, DN1 doesn't.
            res.codes[res.len] = DN2->codes[k];
            res.sum_count += DN2->codes[k].count;
            res.len++;
            k++;
        } else if (DN2->codes[k].pre > DN1->codes[j].pre) {
            // DN1 has it, skip.
            j++;
        } else {
            // Both have it, skip.
            k++; j++;
        }
    }
    // Append remaining from DN2
    while (k < DN2->len) {
        res.codes[res.len] = DN2->codes[k];
        res.sum_count += DN2->codes[k].count;
        res.len++;
        k++;
    }
    return res;
}

static void constructing_pattern_tree(Itemset *Nd, uint32_t *cad_set, uint32_t cad_len, Context *ctx, uint32_t *L1_ranks) {
    // Standard set-enumeration tree recursion
    for (uint32_t i = 0; i < cad_len; i++) {
        uint32_t item_i = cad_set[i];
        
        // In Algorithm, X is parent itemset, Y is subset with last item replaced, P is new itemset.
        // Actually, we need DN1 and DN2 from the parent level.
        // For a fixed prefix, we compare extensions.
        // This is simplified in most implementations by passing the DNs directly.
    }
}

/* --- Simplified but Paper-Compliant Mining --- */

typedef struct Candidate {
    uint32_t item;
    DiffNodeset dn;
    uint32_t support;
} Candidate;

static void mine_recursive(Candidate *parent, Candidate *cands, uint32_t cand_count, Context *ctx) {
    for (uint32_t i = 0; i < cand_count; i++) {
        Candidate *X = &cands[i];
        
        uint32_t next_cand_count = 0;
        Candidate *next_cands = malloc(i * sizeof(Candidate));
        
        for (uint32_t j = 0; j < i; j++) {
            Candidate *Y = &cands[j];
            
            // Theorem 4: DNP = DN2 / DN1
            // If X < Y in rank order (X is current prefix, Y is extension), then DN(XY) = DN(Y) / DN(X)
            DiffNodeset next_dn = set_difference_dn(&Y->dn, &X->dn);
            // Theorem 3: Support(P) = Support(P1) - sum(DN_P)
            // Here P1 is the prefix X.
            uint32_t next_supp = X->support - next_dn.sum_count;
            
            if (next_supp >= ctx->min_sup) {
                next_cands[next_cand_count].item = Y->item;
                next_cands[next_cand_count].dn = next_dn;
                next_cands[next_cand_count].support = next_supp;
                next_cand_count++;
                
                ctx->total_fi++;
            } else {
                free(next_dn.codes);
            }
        }
        
        if (next_cand_count > 0) {
            mine_recursive(X, next_cands, next_cand_count, ctx);
        }
        
        for (uint32_t k = 0; k < next_cand_count; k++) free(next_cands[k].dn.codes);
        free(next_cands);
    }
}

static uint32_t *g_support = NULL;
static int cmp_support_desc(const void *a, const void *b) {
    uint32_t ia = *(uint32_t*)a, ib = *(uint32_t*)b;
    if (g_support[ia] > g_support[ib]) return -1;
    if (g_support[ia] < g_support[ib]) return 1;
    return (ia < ib) ? -1 : 1;
}

static void collect_ns(PPC_Node *n, Nodeset *ns_array, uint32_t *rm) {
    if (!n) return;
    if (n->item != ROOT_ITEM) {
        uint32_t r = rm[n->item];
        ns_array[r].codes[ns_array[r].len++] = (PP_Code){n->pre, n->post, n->count};
    }
    PPC_Node *c = n->children;
    while (c) {
        collect_ns(c, ns_array, rm);
        c = c->sibling;
    }
}

static int cmp_pp(const void *a, const void *b) {
    const PP_Code *pa = (const PP_Code *)a;
    const PP_Code *pb = (const PP_Code *)b;
    return (pa->pre < pb->pre) ? -1 : (pa->pre > pb->pre);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FINPLUS_Params *p = (DM_FINPLUS_Params*)params;
    double ms_val = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_val < 1.0) ? (uint32_t)ceil(ms_val * ds->count) : (uint32_t)ms_val;
    if (min_sup == 0) min_sup = 1;

    printf("[FIN+] Starting. Min Support: %u\n", min_sup);

    // 1. Scan once to find F1
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple*)ds->payload;
    for (size_t i = 0; i < ds->count; i++)
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    
    uint32_t freq_cnt = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_cnt++;
    if (freq_cnt == 0) { free(counts); return DM_SUCCESS; }

    // 2. Sort F1 in descending order as L1 (Procedure Construct_PPC-tree)
    uint32_t *L1 = malloc(freq_cnt * sizeof(uint32_t));
    uint32_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) L1[idx++] = i;
    g_support = counts;
    qsort(L1, freq_cnt, sizeof(uint32_t), cmp_support_desc);
    
    uint32_t *rank_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) rank_map[i] = 0xFFFFFFFF;
    for (uint32_t i = 0; i < freq_cnt; i++) rank_map[L1[i]] = i;

    // 3. Construct PPC-tree
    Pool pool = {0};
    PPC_Node *root = alloc_ppc_node(&pool);
    root->item = ROOT_ITEM;

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *path = malloc(data[i].count * sizeof(uint32_t));
        uint32_t plen = 0;
        for (size_t j = 0; j < data[i].count; j++)
            if (rank_map[data[i].items[j]] != 0xFFFFFFFF) path[plen++] = data[i].items[j];
        if (plen > 0) {
            // Sort path by L1 order (descending support)
            for (uint32_t j = 0; j < plen; j++) {
                for (uint32_t k = j + 1; k < plen; k++) {
                    if (rank_map[path[j]] > rank_map[path[k]]) {
                        uint32_t tmp = path[j]; path[j] = path[k]; path[k] = tmp;
                    }
                }
            }
            PPC_Node *curr = root;
            for (uint32_t k = 0; k < plen; k++) {
                PPC_Node *child = curr->children, *prev = NULL;
                while (child && child->item != path[k]) { prev = child; child = child->sibling; }
                if (!child) {
                    child = alloc_ppc_node(&pool);
                    child->item = path[k];
                    child->parent = curr;
                    if (prev) prev->sibling = child; else curr->children = child;
                }
                child->count++;
                curr = child;
            }
        }
        free(path);
    }

    // 4. Generate pre-order and post-order
    g_pre = 0; g_post = 0;
    traverse_ppc(root);

    // 5. Generate Nodesets for F1 (Algorithm 1, Line 4-5)
    Nodeset *item_nodesets = calloc(freq_cnt, sizeof(Nodeset));
    for (uint32_t i = 0; i < freq_cnt; i++) {
        item_nodesets[i].codes = malloc(counts[L1[i]] * sizeof(PP_Code)); // Upper bound
        item_nodesets[i].len = 0;
        item_nodesets[i].support = counts[L1[i]];
    }
    
    // Scan tree (pre-order visit already done by traverse, but we need to collect)
    collect_ns(root, item_nodesets, rank_map);

    // Sort Nodeset codes by pre-order (Definition 4)
    for (uint32_t i = 0; i < freq_cnt; i++)
        qsort(item_nodesets[i].codes, item_nodesets[i].len, sizeof(PP_Code), cmp_pp);

    // 6. Mine Frequent 2-itemsets (Algorithm 1, Line 7-13)
    Context ctx = { .min_sup = min_sup, .total_fi = freq_cnt, .total_footprint = freq_cnt };
    Candidate *F2_cands = malloc(freq_cnt * sizeof(Candidate));
    uint32_t F2_count = 0;
    
    // We process items in reverse order of L1 for set-enumeration? 
    // The paper says i1 < i2 means i2 is ahead in L1.
    // So if L1 = {e, d, c, b, a}, then a < b < c < d < e.
    // 2-itemsets ixiy where x < y.
    for (int i = 0; i < (int)freq_cnt; i++) {
        uint32_t next_f2_count = 0;
        Candidate *next_f2 = malloc(i * sizeof(Candidate));
        
        for (int j = 0; j < i; j++) {
            // ix = L1[i], iy = L1[j]. Since j < i, iy is ahead of ix, so ix < iy.
            DiffNodeset dn = build_2_itemset_dn(&item_nodesets[i], &item_nodesets[j]);
            uint32_t supp = item_nodesets[i].support - dn.sum_count;
            
            if (supp >= min_sup) {
                next_f2[next_f2_count].item = L1[j];
                next_f2[next_f2_count].dn = dn;
                next_f2[next_f2_count].support = supp;
                next_f2_count++;
                ctx.total_fi++;
            } else {
                free(dn.codes);
            }
        }
        
        // 7. Mine frequent k-itemsets (k >= 3)
        if (next_f2_count > 0) {
            Candidate parent; parent.item = L1[i];
            mine_recursive(&parent, next_f2, next_f2_count, &ctx);
        }
        
        for (uint32_t k = 0; k < next_f2_count; k++) free(next_f2[k].dn.codes);
        free(next_f2);
    }

    printf("[FIN+] Complete. FIs found: %zu\n", ctx.total_fi);
    dm_bench_record_results(ctx.total_fi, ctx.total_footprint);

    for (uint32_t i = 0; i < freq_cnt; i++) free(item_nodesets[i].codes);
    free(item_nodesets); free(F2_cands); free(L1); free(rank_map); free(counts); free_pool(&pool);
    return DM_SUCCESS;
}

static DM_Algorithm algo_finplus = {
    .id = "finplus", .name = "FIN+ Algorithm",
    .description = "Fast Mining of Frequent Itemsets using DiffNodesets (Zhi-Hong Deng).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo_finplus)
