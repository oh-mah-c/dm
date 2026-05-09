#include "algorithms/prepost.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#define ROOT_ITEM 0xFFFFFFFF

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
    size_t count;
    size_t capacity;
} N_List;

typedef struct {
    uint32_t item;
    uint32_t support;
} Header_Entry;

typedef struct {
    uint32_t *items;
    int len;
    N_List nlist;
} Itemset;

typedef struct {
    uint32_t min_sup;
    size_t total_frequent;
    size_t total_footprint;
} PrePost_Context;

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
        np->capacity = 8192;
        np->nodes = (PPC_Node *)malloc(sizeof(PPC_Node) * np->capacity);
        np->used = 0;
        np->next = g_ppc_alloc.head;
        g_ppc_alloc.head = np;
    }
    PPC_Node *n = &g_ppc_alloc.head->nodes[g_ppc_alloc.head->used++];
    n->item = item; n->count = 0; n->pre = 0; n->post = 0; n->parent = parent;
    n->first_child = n->next_sibling = NULL;
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

static void nlist_init(N_List *nl) {
    nl->codes = NULL; nl->count = 0; nl->capacity = 0;
}

static void nlist_add(N_List *nl, uint32_t pre, uint32_t post, uint32_t count) {
    if (nl->count >= nl->capacity) {
        nl->capacity = nl->capacity ? nl->capacity * 2 : 4;
        nl->codes = (PP_Code *)realloc(nl->codes, sizeof(PP_Code) * nl->capacity);
    }
    nl->codes[nl->count].pre = pre;
    nl->codes[nl->count].post = post;
    nl->codes[nl->count].count = count;
    nl->count++;
}

static void nlist_free(N_List *nl) {
    if (nl->codes) free(nl->codes);
    nl->codes = NULL; nl->count = nl->capacity = 0;
}

static uint32_t nlist_support(const N_List *nl) {
    uint32_t s = 0;
    for (size_t i = 0; i < nl->count; i++) s += nl->codes[i].count;
    return s;
}

// nl1 MUST be the ancestor item N-list, nl2 MUST be the descendant item N-list
static N_List nl_intersection(const N_List *nl1, const N_List *nl2) {
    N_List res;
    nlist_init(&res);
    size_t i = 0, j = 0;
    while (i < nl1->count && j < nl2->count) {
        if (nl1->codes[i].pre < nl2->codes[j].pre) {
            if (nl1->codes[i].post > nl2->codes[j].post) {
                if (res.count > 0 && res.codes[res.count - 1].pre == nl1->codes[i].pre) {
                    res.codes[res.count - 1].count += nl2->codes[j].count;
                } else {
                    nlist_add(&res, nl1->codes[i].pre, nl1->codes[i].post, nl2->codes[j].count);
                }
                j++;
            } else i++;
        } else j++;
    }
    return res;
}

static void dfs_codes(PPC_Node *n, uint32_t *pre_counter, uint32_t *post_counter) {
    n->pre = (*pre_counter)++;
    PPC_Node *child = n->first_child;
    while (child) {
        dfs_codes(child, pre_counter, post_counter);
        child = child->next_sibling;
    }
    n->post = (*post_counter)++;
}

static void mining_L(Itemset *Lk, size_t Lk_size, PrePost_Context *ctx) {
    // Lk is assumed to be sorted such that items at smaller indices are MORE frequent
    for (int i = (int)Lk_size - 1; i >= 1; i--) {
        size_t next_size = 0;
        Itemset *next_Lk = (Itemset *)malloc(i * sizeof(Itemset));
        
        // Items j < i are MORE frequent than item i
        for (int j = 0; j < i; j++) {
            // j is ancestor, i is descendant
            N_List combined_nl = nl_intersection(&Lk[j].nlist, &Lk[i].nlist);
            uint32_t support = nlist_support(&combined_nl);
            
            if (support >= ctx->min_sup) {
                Itemset *it = &next_Lk[next_size++];
                it->len = Lk[i].len + 1;
                it->items = (uint32_t *)malloc(it->len * sizeof(uint32_t));
                it->items[0] = Lk[j].items[0];
                memcpy(it->items + 1, Lk[i].items, Lk[i].len * sizeof(uint32_t));
                it->nlist = combined_nl;
                
                ctx->total_frequent++;
                ctx->total_footprint += it->len;
            } else {
                nlist_free(&combined_nl);
            }
        }
        
        if (next_size > 0) {
            // next_Lk is currently sorted by rank of Lk[j], which is 0, 1, 2...
            // header[0] is most frequent, so next_Lk[0] starts with the MOST frequent extension.
            // This order matches the expectation of mining_L.
            mining_L(next_Lk, next_size, ctx);
        }
        
        for (size_t k = 0; k < next_size; k++) {
            free(next_Lk[k].items);
            nlist_free(&next_Lk[k].nlist);
        }
        free(next_Lk);
    }
}

static int cmp_support_desc(const void *a, const void *b) {
    const Header_Entry *ha = (const Header_Entry *)a, *hb = (const Header_Entry *)b;
    if (ha->support > hb->support) return -1;
    if (ha->support < hb->support) return 1;
    return (ha->item < hb->item) ? -1 : 1;
}

static DM_Status run_prepost(DM_Dataset *ds, void *params) {
    DM_PrePost_Params *p = (DM_PrePost_Params *)params;
    double min_support_val = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_support_val < 1.0) ? (uint32_t)ceil(min_support_val * ds->count) : (uint32_t)min_support_val;
    if (min_sup == 0) min_sup = 1;

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++)
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;

    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_count++;
    if (freq_count == 0) { free(counts); return DM_SUCCESS; }

    Header_Entry *header = (Header_Entry *)malloc(sizeof(Header_Entry) * freq_count);
    uint32_t *rank_map = (uint32_t *)malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) rank_map[i] = ROOT_ITEM;
    size_t h_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++)
        if (counts[i] >= min_sup) { header[h_idx].item = i; header[h_idx].support = counts[i]; h_idx++; }
    free(counts);
    qsort(header, freq_count, sizeof(Header_Entry), cmp_support_desc);
    for (size_t i = 0; i < freq_count; i++) rank_map[header[i].item] = (uint32_t)i;

    ppc_alloc_init();
    PPC_Node *root = alloc_ppc_node(ROOT_ITEM, NULL);
    uint32_t *filtered = (uint32_t *)malloc(sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        int f_count = 0;
        for (size_t j = 0; j < data[i].count; j++)
            if (rank_map[data[i].items[j]] != ROOT_ITEM) filtered[f_count++] = data[i].items[j];
        for (int j = 1; j < f_count; j++) {
            uint32_t key = filtered[j]; int k = j - 1;
            while (k >= 0 && rank_map[filtered[k]] > rank_map[key]) { filtered[k+1] = filtered[k]; k--; }
            filtered[k+1] = key;
        }
        PPC_Node *curr = root;
        for (int j = 0; j < f_count; j++) {
            uint32_t item = filtered[j];
            PPC_Node *child = curr->first_child, *prev = NULL;
            while (child) { if (child->item == item) break; prev = child; child = child->next_sibling; }
            if (child) { child->count++; curr = child; }
            else {
                PPC_Node *n = alloc_ppc_node(item, curr); n->count = 1;
                if (prev) prev->next_sibling = n; else curr->first_child = n;
                curr = n;
            }
        }
    }
    free(filtered);
    uint32_t pre = 0, post = 0; dfs_codes(root, &pre, &post);

    Itemset *L1 = (Itemset *)malloc(freq_count * sizeof(Itemset));
    for (size_t i = 0; i < freq_count; i++) {
        L1[i].len = 1; L1[i].items = (uint32_t *)malloc(sizeof(uint32_t)); L1[i].items[0] = header[i].item;
        nlist_init(&L1[i].nlist);
    }
    PPCNodePool *pool = g_ppc_alloc.head;
    while (pool) {
        for (size_t i = 0; i < pool->used; i++) {
            PPC_Node *n = &pool->nodes[i];
            if (n->item != ROOT_ITEM) nlist_add(&L1[rank_map[n->item]].nlist, n->pre, n->post, n->count);
        }
        pool = pool->next;
    }
    for (size_t i = 0; i < freq_count; i++) {
        for (size_t j = 1; j < L1[i].nlist.count; j++) {
            PP_Code key = L1[i].nlist.codes[j]; int k = (int)j - 1;
            while (k >= 0 && L1[i].nlist.codes[k].pre > key.pre) { L1[i].nlist.codes[k+1] = L1[i].nlist.codes[k]; k--; }
            L1[i].nlist.codes[k+1] = key;
        }
    }

    PrePost_Context ctx; ctx.min_sup = min_sup; ctx.total_frequent = freq_count; ctx.total_footprint = freq_count;
    // L1 is sorted most frequent first. header[0] is at index 0.
    mining_L(L1, freq_count, &ctx);

    printf("[PrePost] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);
    printf("[PrePost] Complete. Total frequent itemsets: %zu\n", ctx.total_frequent);
    dm_bench_record_results(ctx.total_frequent, ctx.total_footprint);

    for (size_t i = 0; i < freq_count; i++) { free(L1[i].items); nlist_free(&L1[i].nlist); }
    free(L1); free(header); free(rank_map); ppc_alloc_free();
    return DM_SUCCESS;
}

static DM_Algorithm algo_prepost = {
    .id = "prepost", .name = "PrePost Algorithm",
    .description = "Fast Mining of Frequent Itemsets using N-lists (Deng et al., 2012).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run_prepost
};
DM_REGISTER_ALGORITHM(algo_prepost)
