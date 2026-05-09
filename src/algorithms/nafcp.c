#include "algorithms/nafcp.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define ROOT_ITEM 0xFFFFFFFF

typedef struct {
    uint32_t pre;
    uint32_t post;
    uint32_t count;
} NNode;

typedef struct {
    NNode *nodes;
    size_t size;
    uint32_t support;
} NList;

typedef struct {
    uint32_t *itemset;
    size_t length;
    NList nl;
} Element;

typedef struct FCINode {
    uint32_t *itemset;
    size_t length;
    uint32_t support;
    struct FCINode *next;
} FCINode;

typedef struct {
    uint32_t min_sup;
    FCINode **hash_table;
    size_t hash_size;
    size_t total_closed;
    size_t total_footprint;
} NAFCP_Context;

typedef struct PPCNode {
    uint32_t item;
    uint32_t count;
    uint32_t pre;
    uint32_t post;
    struct PPCNode *children;
    struct PPCNode *next_sibling;
} PPCNode;

static PPCNode* create_ppc_node(uint32_t item) {
    PPCNode *n = calloc(1, sizeof(PPCNode));
    n->item = item; return n;
}

static void free_ppc_tree(PPCNode *n) {
    if (!n) return;
    PPCNode *c = n->children;
    while (c) { PPCNode *next = c->next_sibling; free_ppc_tree(c); c = next; }
    free(n);
}

static uint32_t g_pre = 1, g_post = 1;
static void traverse(PPCNode *n) {
    n->pre = g_pre++;
    for (PPCNode *c = n->children; c; c = c->next_sibling) traverse(c);
    n->post = g_post++;
}

static void build_nl(PPCNode *n, NList *lists, uint32_t *map) {
    if (n->item != ROOT_ITEM) {
        uint32_t idx = map[n->item];
        if (idx != ROOT_ITEM) {
            NList *nl = &lists[idx];
            nl->nodes[nl->size].pre = n->pre;
            nl->nodes[nl->size].post = n->post;
            nl->nodes[nl->size].count = n->count;
            nl->size++; nl->support += n->count;
        }
    }
    for (PPCNode *c = n->children; c; c = c->next_sibling) build_nl(c, lists, map);
}

static bool is_subset_nl(NList *n1, NList *n2) {
    if (n1->support > n2->support) return false;
    size_t i = 0, j = 0;
    while (j < n1->size && i < n2->size) {
        if (n2->nodes[i].pre < n1->nodes[j].pre && n2->nodes[i].post > n1->nodes[j].post) {
            j++;
        } else {
            i++;
        }
    }
    return j == n1->size;
}

static NList intersect_nl(NList *ps1, NList *ps2, uint32_t threshold) {
    NList res = {NULL, 0, 0};
    uint32_t rem_sup1 = ps1->support;
    uint32_t rem_sup2 = ps2->support;
    
    size_t i = 0, j = 0;
    size_t cap = ps2->size;
    res.nodes = malloc(cap * sizeof(NNode));
    
    while (i < ps1->size && j < ps2->size) {
        if (res.support + (rem_sup1 < rem_sup2 ? rem_sup1 : rem_sup2) < threshold) {
            free(res.nodes); res.nodes = NULL; res.size = 0; res.support = 0; return res;
        }
        
        if (ps1->nodes[i].pre < ps2->nodes[j].pre) {
            if (ps1->nodes[i].post > ps2->nodes[j].post) {
                if (res.size > 0 && res.nodes[res.size - 1].pre == ps1->nodes[i].pre) {
                    res.nodes[res.size - 1].count += ps2->nodes[j].count;
                } else {
                    res.nodes[res.size] = ps1->nodes[i];
                    res.nodes[res.size].count = ps2->nodes[j].count;
                    res.size++;
                }
                res.support += ps2->nodes[j].count;
                rem_sup2 -= ps2->nodes[j].count;
                j++;
            } else {
                rem_sup1 -= ps1->nodes[i].count;
                i++;
            }
        } else {
            rem_sup2 -= ps2->nodes[j].count;
            j++;
        }
    }
    return res;
}

static bool is_subset_items(uint32_t *sub, size_t sl, uint32_t *sup, size_t spl) {
    size_t i = 0, j = 0;
    while (i < sl && j < spl) {
        if (sub[i] == sup[j]) i++;
        else if (sub[i] < sup[j]) return false;
        j++;
    }
    return i == sl;
}

static void add_fci(uint32_t *items, size_t len, uint32_t sup, NAFCP_Context *ctx) {
    size_t h = sup % ctx->hash_size;
    
    FCINode *curr = ctx->hash_table[h];
    while (curr) {
        if (curr->support == sup && curr->length >= len && is_subset_items(items, len, curr->itemset, curr->length)) {
            return;
        }
        curr = curr->next;
    }
    
    FCINode **prev = &ctx->hash_table[h];
    curr = *prev;
    while (curr) {
        if (curr->support == sup && len >= curr->length && is_subset_items(curr->itemset, curr->length, items, len)) {
            FCINode *tmp = curr; *prev = curr->next; curr = curr->next;
            ctx->total_closed--; ctx->total_footprint -= tmp->length;
            free(tmp->itemset); free(tmp); continue;
        }
        prev = &curr->next; curr = curr->next;
    }
    
    FCINode *n = malloc(sizeof(FCINode));
    n->itemset = malloc(len * sizeof(uint32_t));
    memcpy(n->itemset, items, len * sizeof(uint32_t));
    n->length = len; n->support = sup;
    n->next = ctx->hash_table[h]; ctx->hash_table[h] = n;
    ctx->total_closed++; ctx->total_footprint += len;
}

static uint32_t* union_items(uint32_t *a, size_t len_a, uint32_t *b, size_t len_b, size_t *len_out) {
    uint32_t *out = malloc((len_a + len_b) * sizeof(uint32_t));
    size_t i = 0, j = 0, k = 0;
    while (i < len_a && j < len_b) {
        if (a[i] < b[j]) out[k++] = a[i++];
        else if (a[i] > b[j]) out[k++] = b[j++];
        else { out[k++] = a[i++]; j++; }
    }
    while (i < len_a) out[k++] = a[i++];
    while (j < len_b) out[k++] = b[j++];
    *len_out = k;
    return out;
}

static void find_fci_rec(Element *Is, size_t num_Is, NAFCP_Context *ctx) {
    bool *rem = calloc(num_Is, sizeof(bool));
    
    for (int i = (int)num_Is - 1; i >= 0; i--) {
        if (rem[i]) continue;
        
        Element *next_Is = malloc(num_Is * sizeof(Element));
        size_t next_cnt = 0;
        
        for (int j = i - 1; j >= 0; j--) {
            if (rem[j]) continue;
            
            bool s_ij = is_subset_nl(&Is[i].nl, &Is[j].nl); 
            
            if (s_ij) {
                if (Is[i].nl.support == Is[j].nl.support) {
                    uint32_t *new_is = union_items(Is[i].itemset, Is[i].length, Is[j].itemset, Is[j].length, &Is[i].length);
                    free(Is[i].itemset); Is[i].itemset = new_is;
                    
                    for (size_t k = 0; k < next_cnt; k++) {
                        uint32_t *new_next = union_items(next_Is[k].itemset, next_Is[k].length, Is[j].itemset, Is[j].length, &next_Is[k].length);
                        free(next_Is[k].itemset); next_Is[k].itemset = new_next;
                    }
                    
                    free(Is[j].itemset); Is[j].itemset = NULL;
                    if (Is[j].nl.nodes) { free(Is[j].nl.nodes); Is[j].nl.nodes = NULL; }
                    rem[j] = true;
                } else {
                    uint32_t *new_is = union_items(Is[i].itemset, Is[i].length, Is[j].itemset, Is[j].length, &Is[i].length);
                    free(Is[i].itemset); Is[i].itemset = new_is;
                    
                    for (size_t k = 0; k < next_cnt; k++) {
                        uint32_t *new_next = union_items(next_Is[k].itemset, next_Is[k].length, Is[j].itemset, Is[j].length, &next_Is[k].length);
                        free(next_Is[k].itemset); next_Is[k].itemset = new_next;
                    }
                    continue;
                }
            } else {
                NList res = intersect_nl(&Is[j].nl, &Is[i].nl, ctx->min_sup);
                if (res.support >= ctx->min_sup) {
                    next_Is[next_cnt].itemset = union_items(Is[i].itemset, Is[i].length, Is[j].itemset, Is[j].length, &next_Is[next_cnt].length);
                    next_Is[next_cnt].nl = res;
                    next_cnt++;
                } else {
                    if (res.nodes) free(res.nodes);
                }
            }
        }
        
        add_fci(Is[i].itemset, Is[i].length, Is[i].nl.support, ctx);
        
        if (next_cnt > 0) {
            for (size_t l = 0; l < next_cnt / 2; l++) {
                Element tmp = next_Is[l];
                next_Is[l] = next_Is[next_cnt - 1 - l];
                next_Is[next_cnt - 1 - l] = tmp;
            }
            find_fci_rec(next_Is, next_cnt, ctx);
        }
        
        for (size_t k = 0; k < next_cnt; k++) {
            if (next_Is[k].itemset) free(next_Is[k].itemset);
            if (next_Is[k].nl.nodes) free(next_Is[k].nl.nodes);
        }
        free(next_Is);
    }
    free(rem);
}

static uint32_t *g_cnts = NULL;
static int cmp_freq(const void *a, const void *b) {
    uint32_t x = *(uint32_t*)a, y = *(uint32_t*)b;
    if (g_cnts[x] > g_cnts[y]) return -1;
    if (g_cnts[x] < g_cnts[y]) return 1;
    return (x < y) ? -1 : 1;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_NAFCP_Params *p = params; double ms_p = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_p < 1.0) ? (uint32_t)ceil(ms_p * ds->count) : (uint32_t)ms_p;
    if (min_sup == 0) min_sup = 1;
    printf("[NAFCP] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);
    DM_Trans_Simple *data = ds->payload;
    
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }
    g_cnts = counts;
    
    uint32_t *freq = malloc((ds->max_id + 1) * sizeof(uint32_t));
    size_t f_cnt = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq[f_cnt++] = i;
    }
    
    if (f_cnt == 0) { free(counts); free(freq); return DM_SUCCESS; }
    
    qsort(freq, f_cnt, sizeof(uint32_t), cmp_freq);
    
    uint32_t *map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    uint32_t *inv = malloc(f_cnt * sizeof(uint32_t));
    memset(map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < f_cnt; i++) { 
        map[freq[i]] = (uint32_t)i; 
        inv[i] = freq[i]; 
    }
    
    PPCNode *root = create_ppc_node(ROOT_ITEM);
    uint32_t *t_items = malloc((ds->max_id + 1) * sizeof(uint32_t));
    size_t *nl_sizes = calloc(f_cnt, sizeof(size_t));
    
    for (size_t i = 0; i < ds->count; i++) {
        size_t tl = 0; 
        for (size_t j = 0; j < data[i].count; j++) {
            if (counts[data[i].items[j]] >= min_sup) t_items[tl++] = data[i].items[j];
        }
        if (tl > 0) {
            qsort(t_items, tl, sizeof(uint32_t), cmp_freq);
            PPCNode *curr = root;
            for (size_t j = 0; j < tl; j++) {
                uint32_t it = t_items[j]; 
                PPCNode *ch = curr->children, *pr = NULL;
                while (ch) { if (ch->item == it) break; pr = ch; ch = ch->next_sibling; }
                if (!ch) { 
                    ch = create_ppc_node(it); 
                    if (pr) pr->next_sibling = ch; else curr->children = ch; 
                    nl_sizes[map[it]]++; 
                }
                ch->count++; 
                curr = ch; 
            }
        }
    }
    
    g_pre = g_post = 1; 
    traverse(root);
    
    NList *nlists = malloc(f_cnt * sizeof(NList));
    for (size_t i = 0; i < f_cnt; i++) { 
        nlists[i].nodes = malloc(nl_sizes[i] * sizeof(NNode)); 
        nlists[i].size = 0; 
        nlists[i].support = 0; 
    }
    build_nl(root, nlists, map); 
    free_ppc_tree(root);
    
    Element *Is = malloc(f_cnt * sizeof(Element));
    for (size_t i = 0; i < f_cnt; i++) { 
        Is[i].length = 1; 
        Is[i].itemset = malloc(sizeof(uint32_t)); 
        Is[i].itemset[0] = inv[i]; 
        Is[i].nl = nlists[i]; 
    }
    
    NAFCP_Context ctx = {min_sup, calloc(10007, sizeof(FCINode*)), 10007, 0, 0};
    find_fci_rec(Is, f_cnt, &ctx);
    
    printf("[NAFCP] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_closed);
    dm_bench_record_results(ctx.total_closed, ctx.total_footprint);
    
    for (size_t i = 0; i < f_cnt; i++) { 
        if (Is[i].itemset) free(Is[i].itemset); 
        if (Is[i].nl.nodes) free(Is[i].nl.nodes); 
    }
    
    for (size_t i = 0; i < ctx.hash_size; i++) {
        FCINode *c = ctx.hash_table[i]; 
        while (c) { 
            FCINode *n = c->next; 
            free(c->itemset); 
            free(c); 
            c = n; 
        }
    }
    free(ctx.hash_table); 
    free(Is); 
    free(nlists); 
    free(nl_sizes); 
    free(t_items); 
    free(map); 
    free(inv); 
    free(freq); 
    free(counts);
    return DM_SUCCESS;
}

static DM_Algorithm algo = { 
    .id = "nafcp", 
    .name = "NAFCP Algorithm", 
    .description = "N-list-based Frequent Closed Pattern Mining", 
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), 
    .run = run 
};
DM_REGISTER_ALGORITHM(algo)
