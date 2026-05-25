#include "algorithms/prepostplus.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

/**
 * PrePost+: An efficient N-lists-based algorithm for mining frequent itemsets.
 * Reference: Zhi-Hong Deng and Sheng-Long Lv, "PrePost+: An efficient N-lists-based algorithm for mining 
 * frequent itemsets via Children-Parent Equivalence pruning", Expert Systems With Applications 42 (2015) 5424-5432.
 */

#define ROOT_ITEM 0xFFFFFFFF

/* --- Data Structures --- */

typedef struct PPC_Node {
    uint32_t item;
    uint32_t count;
    uint32_t pre;
    uint32_t post;
    struct PPC_Node *parent;
    struct PPC_Node *first_child;
    struct PPC_Node *next_sibling;
} PPC_Node;

typedef struct {
    uint32_t pre;
    uint32_t post;
    uint32_t count;
} PP_Code;

typedef struct {
    PP_Code *codes;
    uint32_t len;
} N_List;

typedef struct {
    uint32_t item;
    N_List nlist;
    uint32_t support;
} FI_Node;

typedef struct {
    uint32_t min_sup;
    size_t total_frequent;
    size_t total_footprint;
    uint32_t *rank_map;
    uint32_t *L1_items;
    size_t nf;
} Context;

/* --- Memory Management --- */

typedef struct PPCNodePool {
    PPC_Node *nodes;
    size_t capacity;
    size_t used;
    struct PPCNodePool *next;
} PPCNodePool;

typedef struct {
    PPCNodePool *head;
} PPC_Allocator;

static PPC_Allocator g_ppc_alloc;

static void ppc_alloc_init() { g_ppc_alloc.head = NULL; }

static PPC_Node* alloc_ppc_node(uint32_t item, PPC_Node *parent) {
    if (!g_ppc_alloc.head || g_ppc_alloc.head->used >= g_ppc_alloc.head->capacity) {
        PPCNodePool *np = (PPCNodePool *)malloc(sizeof(PPCNodePool));
        np->capacity = 16384;
        np->nodes = (PPC_Node *)calloc(np->capacity, sizeof(PPC_Node));
        np->used = 0;
        np->next = g_ppc_alloc.head;
        g_ppc_alloc.head = np;
    }
    PPC_Node *n = &g_ppc_alloc.head->nodes[g_ppc_alloc.head->used++];
    n->item = item; n->parent = parent;
    return n;
}

static void ppc_alloc_free() {
    PPCNodePool *curr = g_ppc_alloc.head;
    while (curr) {
        PPCNodePool *next = curr->next;
        free(curr->nodes); free(curr); curr = next;
    }
    g_ppc_alloc.head = NULL;
}

/* --- N-list Utilities --- */

static uint32_t get_support(const N_List *nl) {
    uint32_t s = 0;
    for (uint32_t i = 0; i < nl->len; i++) s += nl->codes[i].count;
    return s;
}

static N_List nl_intersection(const N_List *nl_anc, const N_List *nl_des) {
    N_List res;
    res.codes = malloc(nl_des->len * sizeof(PP_Code));
    res.len = 0;
    
    uint32_t i = 0, j = 0;
    while (i < nl_anc->len && j < nl_des->len) {
        if (nl_anc->codes[i].pre < nl_des->codes[j].pre) {
            if (nl_anc->codes[i].post > nl_des->codes[j].post) {
                if (res.len > 0 && res.codes[res.len - 1].pre == nl_anc->codes[i].pre) {
                    res.codes[res.len - 1].count += nl_des->codes[j].count;
                } else {
                    res.codes[res.len].pre = nl_anc->codes[i].pre;
                    res.codes[res.len].post = nl_anc->codes[i].post;
                    res.codes[res.len].count = nl_des->codes[j].count;
                    res.len++;
                }
                j++;
            } else {
                i++;
            }
        } else {
            j++;
        }
    }
    if (res.len == 0) { free(res.codes); res.codes = NULL; }
    else { res.codes = realloc(res.codes, res.len * sizeof(PP_Code)); }
    return res;
}

/* --- Algorithm 1: Building_Pattern_Tree --- */

static void building_pattern_tree_recursive(FI_Node *Nd, FI_Node *siblings, uint32_t sibling_count, size_t Parent_fit_count, Context *ctx) {
    uint32_t *equiv = malloc(ctx->nf * sizeof(uint32_t));
    uint32_t equiv_count = 0;
    
    FI_Node *children = malloc(sibling_count * sizeof(FI_Node));
    uint32_t child_count = 0;

    for (uint32_t i = 0; i < sibling_count; i++) {
        FI_Node *sibling_i = &siblings[i];
        
        N_List P_nlist = nl_intersection(&sibling_i->nlist, &Nd->nlist);
        uint32_t P_support = get_support(&P_nlist);

        if (P_support == Nd->support) {
            equiv[equiv_count++] = sibling_i->item;
            if (P_nlist.codes) free(P_nlist.codes);
        } else if (P_support >= ctx->min_sup) {
            children[child_count].item = sibling_i->item;
            children[child_count].nlist = P_nlist;
            children[child_count].support = P_support;
            child_count++;
        } else {
            if (P_nlist.codes) free(P_nlist.codes);
        }
    }

    // Line 18-23: Generate frequent itemsets
    size_t num_equiv_subsets = (size_t)1 << equiv_count;
    size_t base_count = (Parent_fit_count == 0) ? 1 : Parent_fit_count;
    size_t Nd_fit_count = num_equiv_subsets * base_count;
    
    ctx->total_frequent += Nd_fit_count;

    for (uint32_t i = 0; i < child_count; i++) {
        building_pattern_tree_recursive(&children[i], children + i + 1, child_count - (i + 1), Nd_fit_count, ctx);
    }

    for (uint32_t i = 0; i < child_count; i++) { if (children[i].nlist.codes) free(children[i].nlist.codes); }
    free(children);
    free(equiv);
}

/* --- Global State for Sort --- */

static uint32_t *g_counts = NULL;
static int cmp_support_asc(const void *a, const void *b) {
    uint32_t ia = *(uint32_t*)a, ib = *(uint32_t*)b;
    if (g_counts[ia] < g_counts[ib]) return -1;
    if (g_counts[ia] > g_counts[ib]) return 1;
    return (ia < ib) ? -1 : 1;
}

static int cmp_support_desc(const void *a, const void *b) {
    uint32_t ia = *(uint32_t*)a, ib = *(uint32_t*)b;
    if (g_counts[ia] > g_counts[ib]) return -1;
    if (g_counts[ia] < g_counts[ib]) return 1;
    return (ia < ib) ? -1 : 1;
}

static uint32_t *g_rank_map = NULL;
static int cmp_rank_asc(const void *a, const void *b) {
    return (int)g_rank_map[*(uint32_t*)a] - (int)g_rank_map[*(uint32_t*)b];
}

static void collect_nlist_asc(PPC_Node *n, N_List *nl_array, uint32_t *rm) {
    if (!n) return;
    if (n->item != ROOT_ITEM) {
        uint32_t r = rm[n->item];
        nl_array[r].codes[nl_array[r].len++] = (PP_Code){n->pre, n->post, n->count};
    }
    PPC_Node *c = n->first_child;
    while (c) { collect_nlist_asc(c, nl_array, rm); c = c->next_sibling; }
}

static void dfs_assign_codes(PPC_Node *n, uint32_t *pre, uint32_t *post) {
    if (!n) return;
    n->pre = (*pre)++;
    PPC_Node *c = n->first_child;
    while (c) {
        dfs_assign_codes(c, pre, post);
        c = c->next_sibling;
    }
    n->post = (*post)++;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_PrePostPlus_Params *p = (DM_PrePostPlus_Params*)params;
    double ms_val = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_val < 1.0) ? (uint32_t)ceil(ms_val * ds->count) : (uint32_t)ms_val;
    if (min_sup == 0) min_sup = 1;

    printf("[PrePost+] Starting. Min Support: %u\n", min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple*)ds->payload;
    for (size_t i = 0; i < ds->count; i++)
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;

    uint32_t freq_cnt = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_cnt++;
    if (freq_cnt == 0) { free(counts); return DM_SUCCESS; }

    uint32_t *L1_desc = malloc(freq_cnt * sizeof(uint32_t));
    uint32_t d_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) L1_desc[d_idx++] = i;
    
    g_counts = counts;
    qsort(L1_desc, freq_cnt, sizeof(uint32_t), cmp_support_desc);

    uint32_t *rank_map_desc = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) rank_map_desc[i] = 0xFFFFFFFF;
    for (uint32_t i = 0; i < freq_cnt; i++) rank_map_desc[L1_desc[i]] = i;
    g_rank_map = rank_map_desc;

    uint32_t *L1_asc = malloc(freq_cnt * sizeof(uint32_t));
    memcpy(L1_asc, L1_desc, freq_cnt * sizeof(uint32_t));
    qsort(L1_asc, freq_cnt, sizeof(uint32_t), cmp_support_asc);

    ppc_alloc_init();
    PPC_Node *root = alloc_ppc_node(ROOT_ITEM, NULL);
    root->count = 0;

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *T = malloc(data[i].count * sizeof(uint32_t));
        uint32_t t_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rank_map_desc[data[i].items[j]] != 0xFFFFFFFF) T[t_len++] = data[i].items[j];
        }
        if (t_len > 0) {
            qsort(T, t_len, sizeof(uint32_t), cmp_rank_asc);
            PPC_Node *curr = root;
            for (uint32_t j = 0; j < t_len; j++) {
                uint32_t item = T[j];
                PPC_Node *child = curr->first_child, *prev = NULL;
                while (child && child->item != item) { prev = child; child = child->next_sibling; }
                if (!child) {
                    child = alloc_ppc_node(item, curr);
                    if (prev) prev->next_sibling = child; else curr->first_child = child;
                }
                child->count++;
                curr = child;
            }
        }
        free(T);
    }

    uint32_t pre = 0, post = 0;
    dfs_assign_codes(root, &pre, &post);

    N_List *f1_nl = calloc(freq_cnt, sizeof(N_List));
    uint32_t *rank_map_asc = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) rank_map_asc[i] = 0xFFFFFFFF;
    for (uint32_t i = 0; i < freq_cnt; i++) {
        rank_map_asc[L1_asc[i]] = i;
        f1_nl[i].codes = malloc(counts[L1_asc[i]] * sizeof(PP_Code));
        f1_nl[i].len = 0;
    }

    collect_nlist_asc(root, f1_nl, rank_map_asc);

    Context ctx = { .min_sup = min_sup, .total_frequent = 0, .total_footprint = 0, .rank_map = rank_map_asc, .L1_items = L1_asc, .nf = freq_cnt };

    FI_Node *f1_nodes = malloc(freq_cnt * sizeof(FI_Node));
    for (uint32_t i = 0; i < freq_cnt; i++) {
        f1_nodes[i].item = L1_asc[i];
        f1_nodes[i].nlist = f1_nl[i];
        f1_nodes[i].support = counts[L1_asc[i]];
    }

    for (uint32_t i = 0; i < freq_cnt; i++) {
        FI_Node *children_of_i = malloc((freq_cnt - i - 1) * sizeof(FI_Node));
        uint32_t child_count = 0;
        
        for (uint32_t j = i + 1; j < freq_cnt; j++) {
            N_List i2 = nl_intersection(&f1_nodes[j].nlist, &f1_nodes[i].nlist);
            uint32_t support = get_support(&i2);
            if (support >= min_sup) {
                children_of_i[child_count].item = L1_asc[j];
                children_of_i[child_count].nlist = i2;
                children_of_i[child_count].support = support;
                child_count++;
            } else {
                if (i2.codes) free(i2.codes);
            }
        }
        
        building_pattern_tree_recursive(&f1_nodes[i], children_of_i, child_count, 0, &ctx);
        
        for (uint32_t k = 0; k < child_count; k++) { if (children_of_i[k].nlist.codes) free(children_of_i[k].nlist.codes); }
        free(children_of_i);
    }

    printf("[PrePost+] Complete. FIs found: %zu\n", ctx.total_frequent);
    dm_bench_record_results(ctx.total_frequent, ctx.total_footprint);

    for (uint32_t i = 0; i < freq_cnt; i++) free(f1_nl[i].codes);
    free(f1_nl); free(f1_nodes); free(L1_asc); free(L1_desc); free(rank_map_asc); free(rank_map_desc); free(counts); ppc_alloc_free();
    return DM_SUCCESS;
}

static DM_Algorithm algo_prepostplus = {
    .id = "prepostplus", .name = "PrePost+ Algorithm",
    .description = "An efficient N-lists-based algorithm for mining frequent itemsets via Children-Parent Equivalence pruning.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo_prepostplus)
