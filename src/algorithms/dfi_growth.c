#include "algorithms/dfi_growth.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

/* --- INTERNAL DATA STRUCTURES --- */

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t support;
} FCI_Itemset;

typedef struct {
    FCI_Itemset *array;
    size_t count;
    size_t capacity;
} FCI_List;

typedef struct DFI_Node {
    uint32_t item;
    uint32_t count; // Maximum support count
    struct DFI_Node *parent;
    struct DFI_Node *first_child;
    struct DFI_Node *next_sibling;
    struct DFI_Node *node_link;
} DFI_Node;

typedef struct {
    uint32_t item;
    uint32_t support; // MSS-I value
    DFI_Node *head;
    DFI_Node *tail;
} DFI_HeaderEntry;

typedef struct {
    DFI_HeaderEntry *entries;
    size_t count;
} DFI_HeaderTable;

typedef struct {
    DFI_Node *root;
    DFI_HeaderTable header_table;
} DFI_Tree;

/* --- MEMORY ALLOCATOR --- */

typedef struct DFI_NodePool {
    DFI_Node *nodes;
    size_t capacity;
    size_t used;
    struct DFI_NodePool *next;
} DFI_NodePool;

typedef struct {
    DFI_NodePool *head;
} DFI_Allocator;

static void alloc_init(DFI_Allocator *alloc) {
    alloc->head = NULL;
}

static DFI_Node* alloc_node(DFI_Allocator *alloc, uint32_t item, DFI_Node *parent) {
    if (!alloc->head || alloc->head->used >= alloc->head->capacity) {
        DFI_NodePool *np = (DFI_NodePool *)malloc(sizeof(DFI_NodePool));
        np->capacity = 4096;
        np->nodes = (DFI_Node *)malloc(sizeof(DFI_Node) * np->capacity);
        np->used = 0;
        np->next = alloc->head;
        alloc->head = np;
    }
    DFI_Node *n = &alloc->head->nodes[alloc->head->used++];
    n->item = item;
    n->count = 0;
    n->parent = parent;
    n->first_child = NULL;
    n->next_sibling = NULL;
    n->node_link = NULL;
    return n;
}

static void alloc_free(DFI_Allocator *alloc) {
    DFI_NodePool *curr = alloc->head;
    while (curr) {
        DFI_NodePool *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
    }
}

/* --- FCI COLLECTION --- */

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static void fci_list_init(FCI_List *list) {
    list->capacity = 1024;
    list->count = 0;
    list->array = (FCI_Itemset *)malloc(sizeof(FCI_Itemset) * list->capacity);
}

static void fci_list_free(FCI_List *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->array[i].items);
    }
    free(list->array);
}

static void fci_list_append(FCI_List *list, const uint32_t *items, size_t count, uint32_t support) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (FCI_Itemset *)realloc(list->array, sizeof(FCI_Itemset) * list->capacity);
    }
    list->array[list->count].items = (uint32_t *)malloc(sizeof(uint32_t) * count);
    memcpy(list->array[list->count].items, items, sizeof(uint32_t) * count);
    qsort(list->array[list->count].items, count, sizeof(uint32_t), cmp_uint32);
    list->array[list->count].count = count;
    list->array[list->count].support = support;
    list->count++;
}

/* --- DFI-GROWTH CORE --- */

static int cmp_support_desc(const void *a, const void *b) {
    const DFI_HeaderEntry *ha = (const DFI_HeaderEntry *)a;
    const DFI_HeaderEntry *hb = (const DFI_HeaderEntry *)b;
    if (ha->support > hb->support) return -1;
    if (ha->support < hb->support) return 1;
    if (ha->item < hb->item) return -1;
    if (ha->item > hb->item) return 1;
    return 0;
}

static DFI_Tree* build_dfi_tree(FCI_List *fcis, uint32_t min_sup, DFI_Allocator *alloc, uint32_t max_id) {
    // Step 1: MSS-I strategy
    uint32_t *item_supports = (uint32_t *)calloc(max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < fcis->count; i++) {
        for (size_t j = 0; j < fcis->array[i].count; j++) {
            uint32_t item = fcis->array[i].items[j];
            if (fcis->array[i].support > item_supports[item]) {
                item_supports[item] = fcis->array[i].support;
            }
        }
    }

    size_t freq_count = 0;
    for (uint32_t i = 0; i <= max_id; i++) {
        if (item_supports[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        free(item_supports);
        return NULL;
    }

    DFI_Tree *tree = (DFI_Tree *)malloc(sizeof(DFI_Tree));
    tree->header_table.count = freq_count;
    tree->header_table.entries = (DFI_HeaderEntry *)malloc(sizeof(DFI_HeaderEntry) * freq_count);
    
    size_t idx = 0;
    for (uint32_t i = 0; i <= max_id; i++) {
        if (item_supports[i] >= min_sup) {
            tree->header_table.entries[idx].item = i;
            tree->header_table.entries[idx].support = item_supports[i];
            tree->header_table.entries[idx].head = NULL;
            tree->header_table.entries[idx].tail = NULL;
            idx++;
        }
    }

    qsort(tree->header_table.entries, freq_count, sizeof(DFI_HeaderEntry), cmp_support_desc);

    uint32_t *rank_map = (uint32_t *)malloc(sizeof(uint32_t) * (max_id + 1));
    memset(rank_map, 0xFF, sizeof(uint32_t) * (max_id + 1));
    DFI_HeaderEntry **entry_map = (DFI_HeaderEntry **)calloc(max_id + 1, sizeof(DFI_HeaderEntry*));
    
    for (size_t i = 0; i < freq_count; i++) {
        rank_map[tree->header_table.entries[i].item] = (uint32_t)i;
        entry_map[tree->header_table.entries[i].item] = &tree->header_table.entries[i];
    }

    tree->root = alloc_node(alloc, 0xFFFFFFFF, NULL);

    uint32_t *sorted_fci = (uint32_t *)malloc(sizeof(uint32_t) * (max_id + 1));
    for (size_t i = 0; i < fcis->count; i++) {
        size_t s_count = 0;
        for (size_t j = 0; j < fcis->array[i].count; j++) {
            uint32_t item = fcis->array[i].items[j];
            if (rank_map[item] != 0xFFFFFFFF) {
                sorted_fci[s_count++] = item;
            }
        }
        
        if (s_count > 0) {
            // Sort items in FCI by rank
            for (size_t j = 1; j < s_count; j++) {
                uint32_t key = sorted_fci[j];
                int k_rank = rank_map[key];
                int p = (int)j - 1;
                while (p >= 0 && (int)rank_map[sorted_fci[p]] > k_rank) {
                    sorted_fci[p+1] = sorted_fci[p];
                    p--;
                }
                sorted_fci[p+1] = key;
            }

            DFI_Node *curr = tree->root;
            for (size_t j = 0; j < s_count; j++) {
                uint32_t item = sorted_fci[j];
                DFI_Node *child = curr->first_child;
                DFI_Node *prev = NULL;
                while (child) {
                    if (child->item == item) break;
                    prev = child;
                    child = child->next_sibling;
                }

                if (child) {
                    // MSR strategy: maximum support replacement
                    if (fcis->array[i].support > child->count) {
                        child->count = fcis->array[i].support;
                    }
                    curr = child;
                } else {
                    DFI_Node *n = alloc_node(alloc, item, curr);
                    n->count = fcis->array[i].support;
                    if (prev) prev->next_sibling = n;
                    else curr->first_child = n;

                    DFI_HeaderEntry *he = entry_map[item];
                    if (he->tail) he->tail->node_link = n;
                    else he->head = n;
                    he->tail = n;
                    curr = n;
                }
            }
        }
    }

    free(sorted_fci);
    free(rank_map);
    free(entry_map);
    free(item_supports);

    return tree;
}

// Conditional database entry
typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t cic; // Conditional Itemset Count
} ConditionalTrans;

typedef struct {
    ConditionalTrans *array;
    size_t count;
    size_t capacity;
    uint32_t max_id;
} ConditionalDB;

static void mine_dfi_tree(DFI_Tree *tree, uint32_t min_sup, size_t k, size_t *total_frequent, size_t *footprint) {
    for (int i = (int)tree->header_table.count - 1; i >= 0; i--) {
        DFI_HeaderEntry *he = &tree->header_table.entries[i];
        
        // Output {alpha U I}
        (*total_frequent)++;
        (*footprint) += k;

        // Construct Conditional Database
        ConditionalDB cond_db;
        cond_db.capacity = 16;
        cond_db.count = 0;
        cond_db.max_id = 0;
        cond_db.array = (ConditionalTrans *)malloc(sizeof(ConditionalTrans) * cond_db.capacity);

        DFI_Node *node = he->head;
        uint32_t *path = (uint32_t *)malloc(sizeof(uint32_t) * 1024);

        while (node) {
            if (node->parent && node->parent->item != 0xFFFFFFFF) {
                size_t p_len = 0;
                DFI_Node *p = node->parent;
                while (p && p->item != 0xFFFFFFFF) {
                    path[p_len++] = p->item;
                    if (p->item > cond_db.max_id) cond_db.max_id = p->item;
                    p = p->parent;
                }
                if (p_len > 0) {
                    if (cond_db.count >= cond_db.capacity) {
                        cond_db.capacity *= 2;
                        cond_db.array = (ConditionalTrans *)realloc(cond_db.array, sizeof(ConditionalTrans) * cond_db.capacity);
                    }
                    cond_db.array[cond_db.count].items = (uint32_t *)malloc(sizeof(uint32_t) * p_len);
                    memcpy(cond_db.array[cond_db.count].items, path, sizeof(uint32_t) * p_len);
                    cond_db.array[cond_db.count].count = p_len;
                    cond_db.array[cond_db.count].cic = node->count;
                    cond_db.count++;
                }
            }
            node = node->node_link;
        }
        free(path);

        if (cond_db.count > 0) {
            // MSS-S strategy: calculate support count of frequent itemset {alpha U I U x}
            // In FP-Growth, we'd just build the tree. Here, the tree construction naturally handles it via MSR.
            // But first we need the MSS-S values for the header table of the conditional tree.
            
            uint32_t *x_supports = (uint32_t *)calloc(cond_db.max_id + 1, sizeof(uint32_t));
            for (size_t j = 0; j < cond_db.count; j++) {
                for (size_t l = 0; l < cond_db.array[j].count; l++) {
                    uint32_t x = cond_db.array[j].items[l];
                    if (cond_db.array[j].cic > x_supports[x]) {
                        x_supports[x] = cond_db.array[j].cic;
                    }
                }
            }

            size_t freq_x = 0;
            for (uint32_t j = 0; j <= cond_db.max_id; j++) {
                if (x_supports[j] >= min_sup) freq_x++;
            }

            if (freq_x > 0) {
                DFI_Allocator local_alloc;
                alloc_init(&local_alloc);
                
                DFI_Tree *cond_tree = (DFI_Tree *)malloc(sizeof(DFI_Tree));
                cond_tree->header_table.count = freq_x;
                cond_tree->header_table.entries = (DFI_HeaderEntry *)malloc(sizeof(DFI_HeaderEntry) * freq_x);
                
                size_t idx_x = 0;
                for (uint32_t j = 0; j <= cond_db.max_id; j++) {
                    if (x_supports[j] >= min_sup) {
                        cond_tree->header_table.entries[idx_x].item = j;
                        cond_tree->header_table.entries[idx_x].support = x_supports[j];
                        cond_tree->header_table.entries[idx_x].head = NULL;
                        cond_tree->header_table.entries[idx_x].tail = NULL;
                        idx_x++;
                    }
                }
                qsort(cond_tree->header_table.entries, freq_x, sizeof(DFI_HeaderEntry), cmp_support_desc);

                uint32_t *rank_x = (uint32_t *)malloc(sizeof(uint32_t) * (cond_db.max_id + 1));
                memset(rank_x, 0xFF, sizeof(uint32_t) * (cond_db.max_id + 1));
                DFI_HeaderEntry **entry_map_x = (DFI_HeaderEntry **)calloc(cond_db.max_id + 1, sizeof(DFI_HeaderEntry*));
                for (size_t j = 0; j < freq_x; j++) {
                    rank_x[cond_tree->header_table.entries[j].item] = (uint32_t)j;
                    entry_map_x[cond_tree->header_table.entries[j].item] = &cond_tree->header_table.entries[j];
                }

                cond_tree->root = alloc_node(&local_alloc, 0xFFFFFFFF, NULL);

                uint32_t *sorted_trans = (uint32_t *)malloc(sizeof(uint32_t) * (cond_db.max_id + 1));
                for (size_t j = 0; j < cond_db.count; j++) {
                    size_t s_c = 0;
                    for (size_t l = 0; l < cond_db.array[j].count; l++) {
                        if (rank_x[cond_db.array[j].items[l]] != 0xFFFFFFFF) {
                            sorted_trans[s_c++] = cond_db.array[j].items[l];
                        }
                    }
                    if (s_c > 0) {
                        // Sort by rank
                        for (size_t l = 1; l < s_c; l++) {
                            uint32_t key = sorted_trans[l];
                            int r = rank_x[key];
                            int p = (int)l - 1;
                            while (p >= 0 && (int)rank_x[sorted_trans[p]] > r) {
                                sorted_trans[p+1] = sorted_trans[p];
                                p--;
                            }
                            sorted_trans[p+1] = key;
                        }

                        DFI_Node *curr = cond_tree->root;
                        for (size_t l = 0; l < s_c; l++) {
                            uint32_t item = sorted_trans[l];
                            DFI_Node *child = curr->first_child;
                            DFI_Node *prev = NULL;
                            while (child) {
                                if (child->item == item) break;
                                prev = child;
                                child = child->next_sibling;
                            }
                            if (child) {
                                if (cond_db.array[j].cic > child->count) child->count = cond_db.array[j].cic;
                                curr = child;
                            } else {
                                DFI_Node *n = alloc_node(&local_alloc, item, curr);
                                n->count = cond_db.array[j].cic;
                                if (prev) prev->next_sibling = n;
                                else curr->first_child = n;
                                DFI_HeaderEntry *he_x = entry_map_x[item];
                                if (he_x->tail) he_x->tail->node_link = n;
                                else he_x->head = n;
                                he_x->tail = n;
                                curr = n;
                            }
                        }
                    }
                }

                mine_dfi_tree(cond_tree, min_sup, k + 1, total_frequent, footprint);

                free(sorted_trans);
                free(rank_x);
                free(entry_map_x);
                free(cond_tree->header_table.entries);
                free(cond_tree);
                alloc_free(&local_alloc);
            }
            free(x_supports);
        }

        for (size_t j = 0; j < cond_db.count; j++) free(cond_db.array[j].items);
        free(cond_db.array);
    }
}

/* --- REUSING CHARM LOGIC TO GET FCIs --- */
// (Adapted from charm.c but returns FCI_List)

typedef struct CFI_Node_Local {
    uint32_t *items;
    size_t num_items;
    uint32_t support;
    uint32_t tid_sum;
    struct CFI_Node_Local *next;
} CFI_Node_Local;

typedef struct {
    uint32_t *items;
    size_t num_items;
    uint32_t *tids;
    size_t num_tids;
    bool deleted;
} CharmNode_Local;

typedef struct {
    CFI_Node_Local **hash_table;
    FCI_List *fci_list;
    uint32_t min_sup;
    uint32_t *pair_counts;
    uint32_t *item_to_idx;
    bool is_root;
    size_t freq_count;
} CharmContext_Local;

static bool is_subset_local(const uint32_t *a, size_t len_a, const uint32_t *b, size_t len_b) {
    if (len_a > len_b) return false;
    size_t i = 0, j = 0;
    while (i < len_a && j < len_b) {
        if (a[i] < b[j]) return false;
        else if (a[i] == b[j]) { i++; j++; }
        else j++;
    }
    return i == len_a;
}

static bool is_subsumed_local(CharmContext_Local *ctx, const uint32_t *items, size_t num_items, uint32_t supp, uint32_t tid_sum) {
    uint32_t hash_idx = tid_sum % 1000003;
    for (CFI_Node_Local *n = ctx->hash_table[hash_idx]; n != NULL; n = n->next) {
        if (n->support == supp && n->tid_sum == tid_sum) {
            if (is_subset_local(items, num_items, n->items, n->num_items)) return true;
        }
    }
    return false;
}

static void add_cfi_local(CharmContext_Local *ctx, const uint32_t *items, size_t num_items, uint32_t supp, uint32_t tid_sum) {
    uint32_t hash_idx = tid_sum % 1000003;
    CFI_Node_Local *n = malloc(sizeof(CFI_Node_Local));
    n->items = malloc(num_items * sizeof(uint32_t));
    memcpy(n->items, items, num_items * sizeof(uint32_t));
    n->num_items = num_items;
    n->support = supp;
    n->tid_sum = tid_sum;
    n->next = ctx->hash_table[hash_idx];
    ctx->hash_table[hash_idx] = n;
    
    fci_list_append(ctx->fci_list, items, num_items, supp);
}

static size_t intersect_tids_local(const uint32_t *tids1, size_t len1, const uint32_t *tids2, size_t len2, uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    while (i < len1 && j < len2) {
        if (tids1[i] < tids2[j]) i++;
        else if (tids1[i] > tids2[j]) j++;
        else { out[k++] = tids1[i]; i++; j++; }
    }
    return k;
}

static void charm_extend_local(CharmNode_Local *nodes, size_t num_nodes, const uint32_t *prefix, size_t prefix_len, CharmContext_Local *ctx) {
    for (size_t i = 0; i < num_nodes; i++) {
        if (nodes[i].deleted) continue;
        uint32_t *curr_prefix = malloc(ctx->freq_count * sizeof(uint32_t));
        if (prefix_len > 0) memcpy(curr_prefix, prefix, prefix_len * sizeof(uint32_t));
        memcpy(curr_prefix + prefix_len, nodes[i].items, nodes[i].num_items * sizeof(uint32_t));
        size_t curr_len = prefix_len + nodes[i].num_items;
        CharmNode_Local *NewN = malloc(num_nodes * sizeof(CharmNode_Local));
        size_t new_count = 0;
        for (size_t j = i + 1; j < num_nodes; j++) {
            if (nodes[j].deleted) continue;
            if (ctx->is_root && ctx->pair_counts) {
                uint32_t idx_i = ctx->item_to_idx[nodes[i].items[0]];
                uint32_t idx_j = ctx->item_to_idx[nodes[j].items[0]];
                size_t p_idx = (idx_i > idx_j) ? ((size_t)idx_i * (idx_i - 1) / 2 + idx_j) : ((size_t)idx_j * (idx_j - 1) / 2 + idx_i);
                if (ctx->pair_counts[p_idx] < ctx->min_sup) continue;
            }
            uint32_t *t_new = malloc((nodes[i].num_tids < nodes[j].num_tids ? nodes[i].num_tids : nodes[j].num_tids) * sizeof(uint32_t));
            size_t t_new_len = intersect_tids_local(nodes[i].tids, nodes[i].num_tids, nodes[j].tids, nodes[j].num_tids, t_new);
            if (t_new_len >= ctx->min_sup) {
                if (t_new_len == nodes[i].num_tids && t_new_len == nodes[j].num_tids) {
                    memcpy(curr_prefix + curr_len, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    curr_len += nodes[j].num_items; nodes[j].deleted = true; free(t_new);
                } else if (t_new_len == nodes[i].num_tids) {
                    memcpy(curr_prefix + curr_len, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    curr_len += nodes[j].num_items; free(t_new);
                } else if (t_new_len == nodes[j].num_tids) {
                    NewN[new_count].items = malloc(nodes[j].num_items * sizeof(uint32_t));
                    memcpy(NewN[new_count].items, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    NewN[new_count].num_items = nodes[j].num_items; NewN[new_count].tids = t_new; NewN[new_count].num_tids = t_new_len; NewN[new_count].deleted = false;
                    new_count++; nodes[j].deleted = true;
                } else {
                    NewN[new_count].items = malloc(nodes[j].num_items * sizeof(uint32_t));
                    memcpy(NewN[new_count].items, nodes[j].items, nodes[j].num_items * sizeof(uint32_t));
                    NewN[new_count].num_items = nodes[j].num_items; NewN[new_count].tids = t_new; NewN[new_count].num_tids = t_new_len; NewN[new_count].deleted = false;
                    new_count++;
                }
            } else free(t_new);
        }
        if (new_count > 0) {
            bool was_root = ctx->is_root; ctx->is_root = false;
            charm_extend_local(NewN, new_count, curr_prefix, curr_len, ctx);
            ctx->is_root = was_root;
            for (size_t x = 0; x < new_count; x++) { free(NewN[x].items); free(NewN[x].tids); }
        }
        free(NewN);
        uint32_t sum = 0; for (size_t x = 0; x < nodes[i].num_tids; x++) sum += nodes[i].tids[x];
        qsort(curr_prefix, curr_len, sizeof(uint32_t), cmp_uint32);
        if (!is_subsumed_local(ctx, curr_prefix, curr_len, nodes[i].num_tids, sum)) add_cfi_local(ctx, curr_prefix, curr_len, nodes[i].num_tids, sum);
        free(curr_prefix);
    }
}

static void get_fcis_with_charm(DM_Dataset *ds, uint32_t min_sup, FCI_List *out_fci) {
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_count++;
    if (freq_count == 0) { free(counts); return; }
    uint32_t *freq_items = malloc(freq_count * sizeof(uint32_t));
    size_t idx = 0; for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_items[idx++] = i;
    uint32_t *item_to_idx = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < freq_count; i++) item_to_idx[freq_items[i]] = (uint32_t)i;
    size_t pair_arr_size = (size_t)freq_count * (freq_count - 1) / 2;
    uint32_t *pair_counts = pair_arr_size > 0 ? calloc(pair_arr_size, sizeof(uint32_t)) : NULL;
    if (pair_counts) {
        for (size_t i = 0; i < ds->count; i++) {
            uint32_t *filtered = malloc(data[i].count * sizeof(uint32_t));
            size_t f_count = 0; for (size_t j = 0; j < data[i].count; j++) if (counts[data[i].items[j]] >= min_sup) filtered[f_count++] = data[i].items[j];
            if (f_count > 1) {
                for (size_t x = 0; x < f_count; x++) {
                    uint32_t idx_x = item_to_idx[filtered[x]];
                    for (size_t y = 0; y < x; y++) {
                        uint32_t idx_y = item_to_idx[filtered[y]];
                        size_t p_idx = (idx_x > idx_y) ? ((size_t)idx_x * (idx_x - 1) / 2 + idx_y) : ((size_t)idx_y * (idx_y - 1) / 2 + idx_x);
                        pair_counts[p_idx]++;
                    }
                }
            }
            free(filtered);
        }
    }
    CharmNode_Local *root_nodes = calloc(freq_count, sizeof(CharmNode_Local));
    for (size_t i = 0; i < freq_count; i++) {
        root_nodes[i].items = malloc(sizeof(uint32_t)); root_nodes[i].items[0] = freq_items[i];
        root_nodes[i].num_items = 1; root_nodes[i].tids = malloc(counts[freq_items[i]] * sizeof(uint32_t));
        root_nodes[i].num_tids = 0; root_nodes[i].deleted = false;
    }
    for (size_t tid = 0; tid < ds->count; tid++) {
        for (size_t j = 0; j < data[tid].count; j++) {
            uint32_t item = data[tid].items[j];
            if (counts[item] >= min_sup) root_nodes[item_to_idx[item]].tids[root_nodes[item_to_idx[item]].num_tids++] = (uint32_t)tid;
        }
    }
    CharmContext_Local ctx; ctx.hash_table = calloc(1000003, sizeof(CFI_Node_Local*)); ctx.fci_list = out_fci; ctx.min_sup = min_sup;
    ctx.pair_counts = pair_counts; ctx.item_to_idx = item_to_idx; ctx.is_root = true; ctx.freq_count = freq_count;
    charm_extend_local(root_nodes, freq_count, NULL, 0, &ctx);
    for (size_t i = 0; i < freq_count; i++) { free(root_nodes[i].items); free(root_nodes[i].tids); }
    free(root_nodes); free(counts); free(freq_items); free(item_to_idx); if (pair_counts) free(pair_counts);
    for (size_t i = 0; i < 1000003; i++) {
        CFI_Node_Local *n = ctx.hash_table[i]; while (n) { CFI_Node_Local *nxt = n->next; free(n->items); free(n); n = nxt; }
    }
    free(ctx.hash_table);
}

/* --- MAIN ENTRY --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_DFI_GROWTH_Params *p = (DM_DFI_GROWTH_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[DFI-Growth] Starting. Step 1: Mining FCIs...\n");
    FCI_List fcis;
    fci_list_init(&fcis);
    get_fcis_with_charm(ds, min_sup, &fcis);
    printf("[DFI-Growth] Found %zu FCIs. Step 2: Deriving FIs using DFI-Growth...\n", fcis.count);

    if (fcis.count == 0) {
        printf("[DFI-Growth] Complete. Total frequent itemsets found: 0\n");
        fci_list_free(&fcis);
        return DM_SUCCESS;
    }

    DFI_Allocator global_alloc;
    alloc_init(&global_alloc);

    DFI_Tree *tree = build_dfi_tree(&fcis, min_sup, &global_alloc, ds->max_id);
    
    size_t total_frequent = 0;
    size_t footprint = 0;

    if (tree) {
        mine_dfi_tree(tree, min_sup, 1, &total_frequent, &footprint);
        free(tree->header_table.entries);
        free(tree);
    }

    alloc_free(&global_alloc);
    fci_list_free(&fcis);

    printf("[DFI-Growth] Complete. Total frequent itemsets found: %zu\n", total_frequent);
    dm_bench_record_results(total_frequent, footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "dfigrowth",
    .name = "DFI-Growth Algorithm",
    .description = "Deriving Frequent Itemsets based on Pattern Growth (Huang et al. 2019).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
