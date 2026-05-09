#include "algorithms/nafcp.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint32_t pre;
    uint32_t post;
    uint32_t count;
} NNode;

typedef struct {
    NNode *nodes;
    size_t size;
    size_t capacity;
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
    PPCNode *node = calloc(1, sizeof(PPCNode));
    node->item = item;
    return node;
}

static void free_ppc_tree(PPCNode *node) {
    if (!node) return;
    PPCNode *child = node->children;
    while (child) {
        PPCNode *next = child->next_sibling;
        free_ppc_tree(child);
        child = next;
    }
    free(node);
}

static uint32_t g_pre_counter = 1;
static uint32_t g_post_counter = 1;
static void traverse_ppc(PPCNode *node) {
    node->pre = g_pre_counter++;
    for (PPCNode *child = node->children; child; child = child->next_sibling) {
        traverse_ppc(child);
    }
    node->post = g_post_counter++;
}

static void build_n_lists(PPCNode *node, NList *n_lists, uint32_t *item_map) {
    if (node->item != (uint32_t)-1) {
        uint32_t mapped = item_map[node->item];
        if (mapped != (uint32_t)-1) {
            NList *nl = &n_lists[mapped];
            if (nl->size == nl->capacity) {
                nl->capacity = nl->capacity == 0 ? 4 : nl->capacity * 2;
                nl->nodes = realloc(nl->nodes, nl->capacity * sizeof(NNode));
            }
            nl->nodes[nl->size].pre = node->pre;
            nl->nodes[nl->size].post = node->post;
            nl->nodes[nl->size].count = node->count;
            nl->size++;
            nl->support += node->count;
        }
    }
    for (PPCNode *child = node->children; child; child = child->next_sibling) {
        build_n_lists(child, n_lists, item_map);
    }
}

static bool N_list_subset(NList *N1, NList *N2) {
    if (N1->support > N2->support) return false;
    size_t i = 0, j = 0;
    while (i < N1->size && j < N2->size) {
        if (N2->nodes[j].pre <= N1->nodes[i].pre && N2->nodes[j].post >= N1->nodes[i].post) {
            i++;
        } else if (N2->nodes[j].post < N1->nodes[i].post) {
            j++;
        } else {
            return false;
        }
    }
    return i == N1->size;
}

static NList N_list_intersection(NList *N1, NList *N2) {
    NList res = {0};
    size_t i = 0, j = 0;
    while (i < N1->size && j < N2->size) {
        if (N1->nodes[i].pre <= N2->nodes[j].pre && N1->nodes[i].post >= N2->nodes[j].post) {
            if (res.size > 0 && res.nodes[res.size-1].pre == N1->nodes[i].pre) {
                res.nodes[res.size-1].count += N2->nodes[j].count;
            } else {
                if (res.size == res.capacity) {
                    res.capacity = res.capacity ? res.capacity * 2 : 4;
                    res.nodes = realloc(res.nodes, res.capacity * sizeof(NNode));
                }
                res.nodes[res.size].pre = N1->nodes[i].pre;
                res.nodes[res.size].post = N1->nodes[i].post;
                res.nodes[res.size].count = N2->nodes[j].count;
                res.size++;
            }
            res.support += N2->nodes[j].count;
            j++;
        } else if (N1->nodes[i].pre < N2->nodes[j].pre) {
            i++;
        } else {
            j++;
        }
    }
    return res;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    return (ia < ib) ? -1 : (ia > ib);
}

static bool is_subset(uint32_t *sub, size_t sub_len, uint32_t *sup, size_t sup_len) {
    size_t i = 0, j = 0;
    while (i < sub_len && j < sup_len) {
        if (sub[i] == sup[j]) i++;
        else if (sub[i] < sup[j]) return false;
        j++;
    }
    return i == sub_len;
}

static void add_to_FCIs(uint32_t *itemset, size_t length, uint32_t support, NAFCP_Context *ctx) {
    size_t h = support % ctx->hash_size;
    FCINode *curr = ctx->hash_table[h];
    while (curr) {
        if (curr->support == support && curr->length >= length && is_subset(itemset, length, curr->itemset, curr->length)) {
            return;
        }
        curr = curr->next;
    }
    
    FCINode **prev = &ctx->hash_table[h];
    curr = *prev;
    while (curr) {
        if (curr->support == support && length >= curr->length && is_subset(curr->itemset, curr->length, itemset, length)) {
            FCINode *to_free = curr;
            *prev = curr->next;
            curr = curr->next;
            ctx->total_closed--;
            ctx->total_footprint -= to_free->length;
            free(to_free->itemset);
            free(to_free);
            continue;
        }
        prev = &curr->next;
        curr = curr->next;
    }

    FCINode *node = malloc(sizeof(FCINode));
    node->itemset = malloc(length * sizeof(uint32_t));
    memcpy(node->itemset, itemset, length * sizeof(uint32_t));
    qsort(node->itemset, length, sizeof(uint32_t), cmp_uint32);
    node->length = length;
    node->support = support;
    node->next = ctx->hash_table[h];
    ctx->hash_table[h] = node;
    ctx->total_closed++;
    ctx->total_footprint += length;
}

static void Find_FCIs(Element *Is, size_t num_Is, NAFCP_Context *ctx) {
    bool *removed = calloc(num_Is, sizeof(bool));
    for (size_t i = 0; i < num_Is; i++) {
        if (removed[i]) continue;
        
        size_t next_cap = num_Is; // Safe initial capacity
        Element *next_Is = malloc(next_cap * sizeof(Element));
        size_t next_count = 0;
        
        uint32_t *current_items = malloc(Is[i].length * sizeof(uint32_t));
        memcpy(current_items, Is[i].itemset, Is[i].length * sizeof(uint32_t));
        size_t current_len = Is[i].length;

        for (size_t j = i + 1; j < num_Is; j++) {
            if (removed[j]) continue;
            
            bool subset_i_j = N_list_subset(&Is[i].nl, &Is[j].nl);
            bool subset_j_i = N_list_subset(&Is[j].nl, &Is[i].nl);

            if (subset_i_j && subset_j_i) { // T(i) == T(j)
                uint32_t *new_items = realloc(current_items, (current_len + Is[j].length) * sizeof(uint32_t));
                memcpy(new_items + current_len, Is[j].itemset, Is[j].length * sizeof(uint32_t));
                current_items = new_items;
                current_len += Is[j].length;
                removed[j] = true;
            } else if (subset_i_j) { // T(i) subset T(j)
                uint32_t *new_items = realloc(current_items, (current_len + Is[j].length) * sizeof(uint32_t));
                memcpy(new_items + current_len, Is[j].itemset, Is[j].length * sizeof(uint32_t));
                current_items = new_items;
                current_len += Is[j].length;
            } else {
                NList intersect_nl = N_list_intersection(&Is[i].nl, &Is[j].nl);
                if (intersect_nl.support >= ctx->min_sup) {
                    next_Is[next_count].length = Is[i].length + Is[j].length;
                    next_Is[next_count].itemset = malloc(next_Is[next_count].length * sizeof(uint32_t));
                    memcpy(next_Is[next_count].itemset, Is[i].itemset, Is[i].length * sizeof(uint32_t));
                    memcpy(next_Is[next_count].itemset + Is[i].length, Is[j].itemset, Is[j].length * sizeof(uint32_t));
                    next_Is[next_count].nl = intersect_nl;
                    next_count++;
                } else {
                    if (intersect_nl.nodes) free(intersect_nl.nodes);
                }
            }
        }
        
        add_to_FCIs(current_items, current_len, Is[i].nl.support, ctx);
        
        if (next_count > 0) {
            Find_FCIs(next_Is, next_count, ctx);
        }
        
        for (size_t k = 0; k < next_count; k++) {
            free(next_Is[k].itemset);
            if (next_Is[k].nl.nodes) free(next_Is[k].nl.nodes);
        }
        free(next_Is);
        free(current_items);
    }
    free(removed);
}

static uint32_t *g_counts = NULL;
static int cmp_freq_desc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (g_counts[ia] > g_counts[ib]) return -1;
    if (g_counts[ia] < g_counts[ib]) return 1;
    return (ia < ib) ? -1 : 1;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_NAFCP_Params *nafcp_params = (DM_NAFCP_Params *)params;
    double min_sup_param = nafcp_params ? nafcp_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[NAFCP] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++)
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    
    g_counts = counts;
    uint32_t *freq_items = malloc((ds->max_id + 1) * sizeof(uint32_t));
    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_items[freq_count++] = i;
    
    if (freq_count == 0) {
        printf("[NAFCP] Complete. Total frequent closed itemsets found: 0\n");
        dm_bench_record_results(0, 0);
        free(freq_items); free(counts); return DM_SUCCESS;
    }
    
    qsort(freq_items, freq_count, sizeof(uint32_t), cmp_freq_desc);
    
    uint32_t *item_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    uint32_t *inv_map = malloc(freq_count * sizeof(uint32_t));
    memset(item_map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < freq_count; i++) { item_map[freq_items[i]] = (uint32_t)i; inv_map[i] = freq_items[i]; }
    
    PPCNode *root = create_ppc_node((uint32_t)-1);
    uint32_t *t_items = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        size_t t_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) t_items[t_len++] = item;
        }
        if (t_len > 0) {
            qsort(t_items, t_len, sizeof(uint32_t), cmp_freq_desc);
            PPCNode *curr = root;
            for (size_t j = 0; j < t_len; j++) {
                uint32_t item = t_items[j];
                PPCNode *child = curr->children, *prev = NULL;
                while (child) { if (child->item == item) break; prev = child; child = child->next_sibling; }
                if (!child) {
                    child = create_ppc_node(item);
                    if (prev) prev->next_sibling = child; else curr->children = child;
                }
                child->count++; curr = child;
            }
        }
    }
    free(t_items);
    
    g_pre_counter = g_post_counter = 1;
    traverse_ppc(root);
    
    NList *n_lists = calloc(freq_count, sizeof(NList));
    build_n_lists(root, n_lists, item_map);
    free_ppc_tree(root);
    
    Element *Is = malloc(freq_count * sizeof(Element));
    for (size_t i = 0; i < freq_count; i++) {
        Is[i].itemset = malloc(sizeof(uint32_t)); Is[i].itemset[0] = inv_map[i];
        Is[i].length = 1; Is[i].nl = n_lists[i];
    }
    
    NAFCP_Context ctx = {0};
    ctx.min_sup = min_sup;
    ctx.hash_size = 10007;
    ctx.hash_table = calloc(ctx.hash_size, sizeof(FCINode *));
    
    Find_FCIs(Is, freq_count, &ctx);
    
    printf("[NAFCP] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_closed);
    dm_bench_record_results(ctx.total_closed, ctx.total_footprint);
    
    for (size_t i = 0; i < freq_count; i++) { free(Is[i].itemset); if (Is[i].nl.nodes) free(Is[i].nl.nodes); }
    free(Is); free(n_lists);
    for (size_t i = 0; i < ctx.hash_size; i++) {
        FCINode *curr = ctx.hash_table[i];
        while (curr) { FCINode *next = curr->next; free(curr->itemset); free(curr); curr = next; }
    }
    free(ctx.hash_table); free(item_map); free(inv_map); free(freq_items); free(counts);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "nafcp", .name = "NAFCP Algorithm",
    .description = "An N-list-based Algorithm for Mining Frequent Closed Patterns",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo)
