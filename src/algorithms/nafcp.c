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
    struct FCINode *next;
} FCINode;

typedef struct {
    uint32_t min_sup;
    FCINode **hash_table;
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
    free_ppc_tree(node->children);
    free_ppc_tree(node->next_sibling);
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
    for (PPCNode *child = node->children; child; child = child->next_sibling) {
        build_n_lists(child, n_lists, item_map);
    }
}

// Nodelist subset check: T(N1) subset T(N2)
static bool N_list_check(NList *N1, NList *N2) {
    size_t i = 0, j = 0;
    while (i < N1->size && j < N2->size) {
        // Find ancestor of N1[i] in N2
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

// Nodelist intersection: T(N1) intersect T(N2)
static NList N_list_intersection(NList *N1, NList *N2) {
    NList res = {0};
    res.capacity = N1->size + N2->size;
    if (res.capacity > 0) res.nodes = malloc(res.capacity * sizeof(NNode));
    else res.nodes = NULL;
    
    size_t i = 0, j = 0;
    while (i < N1->size && j < N2->size) {
        // One must be ancestor of other
        if (N1->nodes[i].pre <= N2->nodes[j].pre && N1->nodes[i].post >= N2->nodes[j].post) {
            // N1[i] is ancestor of N2[j]
            res.nodes[res.size++] = N2->nodes[j];
            res.support += N2->nodes[j].count;
            j++;
        } else if (N2->nodes[j].pre <= N1->nodes[i].pre && N2->nodes[j].post >= N1->nodes[i].post) {
            // N2[j] is ancestor of N1[i]
            res.nodes[res.size++] = N1->nodes[i];
            res.support += N1->nodes[i].count;
            i++;
        } else if (N1->nodes[i].post < N2->nodes[j].post) {
            i++;
        } else {
            j++;
        }
    }
    return res;
}

static void union_in_place(Element *el, uint32_t *items, size_t len) {
    uint32_t *new_items = malloc((el->length + len) * sizeof(uint32_t));
    size_t i = 0, j = 0, k = 0;
    while (i < el->length && j < len) {
        if (el->itemset[i] < items[j]) new_items[k++] = el->itemset[i++];
        else if (el->itemset[i] > items[j]) new_items[k++] = items[j++];
        else {
            new_items[k++] = el->itemset[i++];
            j++;
        }
    }
    while (i < el->length) new_items[k++] = el->itemset[i++];
    while (j < len) new_items[k++] = items[j++];
    free(el->itemset);
    el->itemset = new_items;
    el->length = k;
}

static uint32_t* union_items(uint32_t *a, size_t len_a, uint32_t *b, size_t len_b, size_t *out_len) {
    uint32_t *new_items = malloc((len_a + len_b) * sizeof(uint32_t));
    size_t i = 0, j = 0, k = 0;
    while (i < len_a && j < len_b) {
        if (a[i] < b[j]) new_items[k++] = a[i++];
        else if (a[i] > b[j]) new_items[k++] = b[j++];
        else {
            new_items[k++] = a[i++];
            j++;
        }
    }
    while (i < len_a) new_items[k++] = a[i++];
    while (j < len_b) new_items[k++] = b[j++];
    *out_len = k;
    return new_items;
}

static bool is_subset(uint32_t *sub, size_t sub_len, uint32_t *sup, size_t sup_len) {
    size_t i = 0, j = 0;
    while (i < sub_len && j < sup_len) {
        if (sub[i] < sup[j]) return false;
        if (sub[i] == sup[j]) i++;
        j++;
    }
    return i == sub_len;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static bool Subsumption_check(uint32_t *itemset, size_t length, uint32_t support, NAFCP_Context *ctx) {
    FCINode *curr = ctx->hash_table[support];
    while (curr) {
        if (curr->length >= length && is_subset(itemset, length, curr->itemset, curr->length)) {
            return true;
        }
        curr = curr->next;
    }
    return false;
}

static void add_to_FCIs(uint32_t *itemset, size_t length, uint32_t support, NAFCP_Context *ctx) {
    uint32_t *sorted = malloc(length * sizeof(uint32_t));
    memcpy(sorted, itemset, length * sizeof(uint32_t));
    qsort(sorted, length, sizeof(uint32_t), cmp_uint32);
    
    if (Subsumption_check(sorted, length, support, ctx)) {
        free(sorted);
        return;
    }
    
    FCINode *node = malloc(sizeof(FCINode));
    node->itemset = sorted;
    node->length = length;
    node->next = ctx->hash_table[support];
    ctx->hash_table[support] = node;
    ctx->total_closed++;
    ctx->total_footprint += length;
}

static void Find_FCIs(Element *Is, size_t num_Is, NAFCP_Context *ctx) {
    bool *removed = calloc(num_Is, sizeof(bool));
    for (size_t i = 0; i < num_Is; i++) {
        if (removed[i]) continue;
        
        Element *FCIs_next = malloc((num_Is - i - 1) * sizeof(Element));
        size_t next_count = 0;
        
        for (size_t j = i + 1; j < num_Is; j++) {
            if (removed[j]) continue;
            
            // CHARM-style subsumption
            if (N_list_check(&Is[j].nl, &Is[i].nl)) { // T(j) subset T(i)
                if (Is[j].nl.support == Is[i].nl.support) { // T(j) == T(i)
                    union_in_place(&Is[i], Is[j].itemset, Is[j].length);
                    removed[j] = true;
                } else {
                    // T(j) subset T(i) => j implies i
                    union_in_place(&Is[j], Is[i].itemset, Is[i].length);
                }
            } else if (N_list_check(&Is[i].nl, &Is[j].nl)) { // T(i) subset T(j)
                // T(i) subset T(j) => i implies j
                union_in_place(&Is[i], Is[j].itemset, Is[j].length);
            }
            
            Element fci;
            fci.itemset = union_items(Is[i].itemset, Is[i].length, Is[j].itemset, Is[j].length, &fci.length);
            fci.nl = N_list_intersection(&Is[i].nl, &Is[j].nl);
            
            if (fci.nl.support >= ctx->min_sup) {
                FCIs_next[next_count++] = fci;
            } else {
                free(fci.itemset);
                free(fci.nl.nodes);
            }
        }
        
        add_to_FCIs(Is[i].itemset, Is[i].length, Is[i].nl.support, ctx);
        
        if (next_count > 0) {
            Find_FCIs(FCIs_next, next_count, ctx);
        }
        
        for (size_t k = 0; k < next_count; k++) {
            free(FCIs_next[k].itemset);
            free(FCIs_next[k].nl.nodes);
        }
        free(FCIs_next);
    }
    free(removed);
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

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_NAFCP_Params *nafcp_params = (DM_NAFCP_Params *)params;
    double min_sup_param = nafcp_params ? nafcp_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[NAFCP] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }
    
    g_counts = counts;
    uint32_t *freq_items = malloc((ds->max_id + 1) * sizeof(uint32_t));
    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            freq_items[freq_count++] = i;
        }
    }
    
    if (freq_count == 0) {
        printf("[NAFCP] Complete. Total frequent closed itemsets found: 0\n");
        dm_bench_record_results(0, 0);
        free(freq_items);
        free(counts);
        return DM_SUCCESS;
    }
    
    qsort(freq_items, freq_count, sizeof(uint32_t), cmp_freq_desc);
    
    uint32_t *item_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    uint32_t *inv_map = malloc(freq_count * sizeof(uint32_t));
    memset(item_map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < freq_count; i++) {
        item_map[freq_items[i]] = i;
        inv_map[i] = freq_items[i];
    }
    
    PPCNode *root = create_ppc_node((uint32_t)-1);
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = malloc(data[i].count * sizeof(uint32_t));
        size_t t_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) {
                t_items[t_len++] = item;
            }
        }
        if (t_len > 0) {
            qsort(t_items, t_len, sizeof(uint32_t), cmp_freq_desc);
            PPCNode *curr = root;
            for (size_t j = 0; j < t_len; j++) {
                uint32_t item = t_items[j];
                PPCNode *child = curr->children;
                PPCNode *prev = NULL;
                while (child != NULL) {
                    if (child->item == item) break;
                    prev = child;
                    child = child->next_sibling;
                }
                if (child == NULL) {
                    child = create_ppc_node(item);
                    if (prev == NULL) curr->children = child;
                    else prev->next_sibling = child;
                }
                child->count++;
                curr = child;
            }
        }
        free(t_items);
    }
    
    g_pre_counter = 1;
    g_post_counter = 1;
    traverse_ppc(root);
    
    NList *n_lists = calloc(freq_count, sizeof(NList));
    build_n_lists(root, n_lists, item_map);
    free_ppc_tree(root);
    
    Element *Is = malloc(freq_count * sizeof(Element));
    for (size_t i = 0; i < freq_count; i++) {
        Is[i].itemset = malloc(sizeof(uint32_t));
        Is[i].itemset[0] = inv_map[i];
        Is[i].length = 1;
        Is[i].nl = n_lists[i];
    }
    
    NAFCP_Context ctx = {0};
    ctx.min_sup = min_sup;
    ctx.hash_table = calloc(ds->count + 1, sizeof(FCINode *));
    
    Find_FCIs(Is, freq_count, &ctx);
    
    printf("[NAFCP] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_closed);
    dm_bench_record_results(ctx.total_closed, ctx.total_footprint);
    
    for (size_t i = 0; i < freq_count; i++) {
        free(Is[i].itemset);
        free(Is[i].nl.nodes);
    }
    free(Is);
    free(n_lists);
    
    for (size_t i = 0; i <= ds->count; i++) {
        FCINode *curr = ctx.hash_table[i];
        while (curr) {
            FCINode *next = curr->next;
            free(curr->itemset);
            free(curr);
            curr = next;
        }
    }
    free(ctx.hash_table);
    free(item_map);
    free(inv_map);
    free(freq_items);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "nafcp",
    .name = "NAFCP Algorithm",
    .description = "An N-list-based Algorithm for Mining Frequent Closed Patterns",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
