#include "algorithms/closetplus.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#define ROOT_ITEM 0xFFFFFFFF

/* --- INTERNAL DATA STRUCTURES --- */

typedef struct FP_Node {
    uint32_t item;
    uint32_t count;
    struct FP_Node *parent;
    struct FP_Node *first_child;
    struct FP_Node *next_sibling;
    struct FP_Node *node_link;
} FP_Node;

typedef struct {
    uint32_t item;
    uint32_t support;
    FP_Node *head;
    FP_Node *tail;
} Header_Entry;

typedef struct {
    Header_Entry *entries;
    size_t count;
} Header_Table;

typedef struct {
    FP_Node *root;
    Header_Table header_table;
    uint32_t max_item_id;
    size_t total_nodes;
    size_t total_counts;
} FP_Tree;

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t weight;
} Weighted_Trans;

typedef struct {
    Weighted_Trans *array;
    size_t count;
    size_t capacity;
    uint32_t max_id;
} Weighted_Trans_List;

/* --- MEMORY POOL --- */
typedef struct NodePool {
    FP_Node *nodes;
    size_t capacity;
    size_t used;
    struct NodePool *next;
} NodePool;

typedef struct {
    NodePool *head;
} FP_Allocator;

static void alloc_init(FP_Allocator *alloc) {
    alloc->head = NULL;
}

static FP_Node* alloc_node(FP_Allocator *alloc, uint32_t item, FP_Node *parent) {
    if (!alloc->head || alloc->head->used >= alloc->head->capacity) {
        NodePool *np = (NodePool *)malloc(sizeof(NodePool));
        np->capacity = 8192;
        np->nodes = (FP_Node *)malloc(sizeof(FP_Node) * np->capacity);
        np->used = 0;
        np->next = alloc->head;
        alloc->head = np;
    }
    FP_Node *n = &alloc->head->nodes[alloc->head->used++];
    n->item = item;
    n->count = 0;
    n->parent = parent;
    n->first_child = NULL;
    n->next_sibling = NULL;
    n->node_link = NULL;
    return n;
}

static void alloc_free(FP_Allocator *alloc) {
    NodePool *curr = alloc->head;
    while (curr) {
        NodePool *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
    }
    alloc->head = NULL;
}

/* --- FREQUENT CLOSED ITEMSETS REGISTRY (For Dense Bottom-Up) --- */
typedef struct {
    uint32_t *items;
    size_t count;
} Closet_Itemset;

typedef struct {
    Closet_Itemset *array;
    size_t count;
    size_t capacity;
} Closet_Itemset_List;

typedef struct {
    Closet_Itemset_List *support_map;
    uint32_t max_support;
    size_t total_fci;
    size_t total_footprint;
} FCI_Registry;

static bool is_subset(const uint32_t *x, size_t x_len, const uint32_t *t, size_t t_len) {
    size_t i = 0, j = 0;
    while (i < x_len && j < t_len) {
        if (x[i] == t[j]) { i++; j++; }
        else if (x[i] > t[j]) { j++; }
        else { return false; }
    }
    return i == x_len;
}

static bool is_subsumed(FCI_Registry *registry, uint32_t *x_items, size_t x_count, uint32_t supp) {
    if (!registry || supp > registry->max_support) return false;
    Closet_Itemset_List *list = &registry->support_map[supp];
    for (size_t i = 0; i < list->count; i++) {
        if (is_subset(x_items, x_count, list->array[i].items, list->array[i].count)) {
            return true;
        }
    }
    return false;
}

static void add_to_registry(FCI_Registry *registry, uint32_t *x_items, size_t x_count, uint32_t supp) {
    if (!registry || supp > registry->max_support) return;
    Closet_Itemset_List *list = &registry->support_map[supp];
    if (list->count >= list->capacity) {
        list->capacity = list->capacity == 0 ? 4 : list->capacity * 2;
        list->array = (Closet_Itemset *)realloc(list->array, sizeof(Closet_Itemset) * list->capacity);
    }
    list->array[list->count].items = x_items;
    list->array[list->count].count = x_count;
    list->count++;
    registry->total_fci++;
    registry->total_footprint += x_count;
}

static bool in_array(uint32_t item, const uint32_t *arr, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (arr[i] == item) return true;
        if (arr[i] > item) break;
    }
    return false;
}

/* --- TREE CONSTRUCTION --- */

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static int cmp_support_desc(const void *a, const void *b) {
    const Header_Entry *ha = (const Header_Entry *)a;
    const Header_Entry *hb = (const Header_Entry *)b;
    if (ha->support > hb->support) return -1;
    if (ha->support < hb->support) return 1;
    if (ha->item < hb->item) return -1;
    if (ha->item > hb->item) return 1;
    return 0;
}

static void sort_transaction(uint32_t *items, size_t count, uint32_t *rank_map) {
    for (size_t i = 1; i < count; i++) {
        uint32_t key = items[i];
        size_t j = i;
        while (j > 0 && rank_map[items[j - 1]] > rank_map[key]) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = key;
    }
}

static FP_Tree* build_tree(Weighted_Trans_List *db, uint32_t min_sup, FP_Allocator *alloc) {
    uint32_t *counts = (uint32_t *)calloc(db->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < db->count; i++) {
        for (size_t j = 0; j < db->array[i].count; j++) {
            counts[db->array[i].items[j]] += db->array[i].weight;
        }
    }

    size_t freq_count = 0;
    for (uint32_t i = 0; i <= db->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        free(counts);
        return NULL;
    }

    FP_Tree *tree = (FP_Tree *)malloc(sizeof(FP_Tree));
    tree->header_table.count = freq_count;
    tree->header_table.entries = (Header_Entry *)malloc(sizeof(Header_Entry) * freq_count);
    tree->total_nodes = 1;
    tree->total_counts = 0;
    
    size_t idx = 0;
    uint32_t local_max_id = 0;
    for (uint32_t i = 0; i <= db->max_id; i++) {
        if (counts[i] >= min_sup) {
            tree->header_table.entries[idx].item = i;
            tree->header_table.entries[idx].support = counts[i];
            tree->header_table.entries[idx].head = NULL;
            tree->header_table.entries[idx].tail = NULL;
            if (i > local_max_id) local_max_id = i;
            idx++;
        }
    }
    tree->max_item_id = local_max_id;

    qsort(tree->header_table.entries, freq_count, sizeof(Header_Entry), cmp_support_desc);

    uint32_t *rank_map = (uint32_t *)calloc(db->max_id + 1, sizeof(uint32_t));
    Header_Entry **entry_map = (Header_Entry **)calloc(db->max_id + 1, sizeof(Header_Entry*));
    
    for (uint32_t i = 0; i <= db->max_id; i++) {
        rank_map[i] = 0xFFFFFFFF;
    }
    for (size_t i = 0; i < freq_count; i++) {
        rank_map[tree->header_table.entries[i].item] = (uint32_t)i;
        entry_map[tree->header_table.entries[i].item] = &tree->header_table.entries[i];
    }

    tree->root = alloc_node(alloc, ROOT_ITEM, NULL);

    uint32_t *filtered = (uint32_t *)malloc(sizeof(uint32_t) * (db->max_id + 1));
    for (size_t i = 0; i < db->count; i++) {
        size_t f_count = 0;
        for (size_t j = 0; j < db->array[i].count; j++) {
            uint32_t item = db->array[i].items[j];
            if (counts[item] >= min_sup) {
                filtered[f_count++] = item;
            }
        }
        if (f_count > 0) {
            sort_transaction(filtered, f_count, rank_map);
            
            FP_Node *curr = tree->root;
            for (size_t j = 0; j < f_count; j++) {
                uint32_t item = filtered[j];
                tree->total_counts += db->array[i].weight;
                FP_Node *child = curr->first_child;
                FP_Node *prev = NULL;
                while (child) {
                    if (child->item == item) break;
                    prev = child;
                    child = child->next_sibling;
                }
                if (child) {
                    child->count += db->array[i].weight;
                    curr = child;
                } else {
                    FP_Node *n = alloc_node(alloc, item, curr);
                    n->count = db->array[i].weight;
                    tree->total_nodes++;
                    if (prev) prev->next_sibling = n;
                    else curr->first_child = n;
                    
                    Header_Entry *he = entry_map[item];
                    if (he->tail) he->tail->node_link = n;
                    else he->head = n;
                    he->tail = n;
                    
                    curr = n;
                }
            }
        }
    }

    free(filtered);
    free(rank_map);
    free(entry_map);
    free(counts);

    return tree;
}

/* --- DFS MINING: DENSE (BOTTOM-UP PHYSICAL PROJECTION) --- */

static void mine_tree_bottom_up(FP_Tree *tree, uint32_t min_sup, uint32_t *prefix, size_t prefix_len, uint32_t prefix_supp, FCI_Registry *registry, FP_Allocator *alloc) {
    size_t y_count = 0;
    for (size_t i = 0; i < tree->header_table.count; i++) {
        if (tree->header_table.entries[i].support == prefix_supp) {
            y_count++;
        }
    }

    uint32_t *x_new = NULL;
    if (prefix_len + y_count > 0) {
        x_new = (uint32_t *)malloc(sizeof(uint32_t) * (prefix_len + y_count));
        if (prefix_len > 0) {
            memcpy(x_new, prefix, sizeof(uint32_t) * prefix_len);
        }
        size_t idx = prefix_len;
        for (size_t i = 0; i < tree->header_table.count; i++) {
            if (tree->header_table.entries[i].support == prefix_supp) {
                x_new[idx++] = tree->header_table.entries[i].item;
            }
        }
        qsort(x_new, prefix_len + y_count, sizeof(uint32_t), cmp_uint32);

        if (!is_subsumed(registry, x_new, prefix_len + y_count, prefix_supp)) {
            add_to_registry(registry, x_new, prefix_len + y_count, prefix_supp);
        } else {
            free(x_new);
        }
    }

    uint32_t *item_to_supp = (uint32_t *)calloc(tree->max_item_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < tree->header_table.count; i++) {
        item_to_supp[tree->header_table.entries[i].item] = tree->header_table.entries[i].support;
    }

    for (int i = tree->header_table.count - 1; i >= 0; i--) {
        Header_Entry *he = &tree->header_table.entries[i];
        if (he->support == prefix_supp) continue;

        uint32_t new_supp = he->support;
        uint32_t *new_prefix = (uint32_t *)malloc(sizeof(uint32_t) * (prefix_len + y_count + 1));
        if (prefix_len + y_count > 0) {
            memcpy(new_prefix, x_new, sizeof(uint32_t) * (prefix_len + y_count));
        }
        new_prefix[prefix_len + y_count] = he->item;
        qsort(new_prefix, prefix_len + y_count + 1, sizeof(uint32_t), cmp_uint32);

        if (is_subsumed(registry, new_prefix, prefix_len + y_count + 1, new_supp)) {
            free(new_prefix);
            continue;
        }

        Weighted_Trans_List cond_db;
        cond_db.capacity = 16;
        cond_db.count = 0;
        cond_db.max_id = 0;
        cond_db.array = (Weighted_Trans *)malloc(sizeof(Weighted_Trans) * cond_db.capacity);

        FP_Node *node = he->head;
        uint32_t *path = (uint32_t *)malloc(sizeof(uint32_t) * 1024);

        while (node) {
            if (node->parent && node->parent->item != ROOT_ITEM) {
                size_t p_len = 0;
                FP_Node *p = node->parent;
                while (p && p->item != ROOT_ITEM) {
                    if (item_to_supp[p->item] < prefix_supp) {
                        path[p_len++] = p->item;
                        if (p->item > cond_db.max_id) cond_db.max_id = p->item;
                    }
                    p = p->parent;
                }
                if (p_len > 0) {
                    if (cond_db.count >= cond_db.capacity) {
                        cond_db.capacity *= 2;
                        cond_db.array = (Weighted_Trans *)realloc(cond_db.array, sizeof(Weighted_Trans) * cond_db.capacity);
                    }
                    cond_db.array[cond_db.count].items = (uint32_t *)malloc(sizeof(uint32_t) * p_len);
                    memcpy(cond_db.array[cond_db.count].items, path, sizeof(uint32_t) * p_len);
                    cond_db.array[cond_db.count].count = p_len;
                    cond_db.array[cond_db.count].weight = node->count;
                    cond_db.count++;
                }
            }
            node = node->node_link;
        }
        free(path);

        if (cond_db.count > 0) {
            FP_Allocator local_alloc;
            alloc_init(&local_alloc);
            
            FP_Tree *cond_tree = build_tree(&cond_db, min_sup, &local_alloc);
            if (cond_tree) {
                mine_tree_bottom_up(cond_tree, min_sup, new_prefix, prefix_len + y_count + 1, new_supp, registry, &local_alloc);
                free(cond_tree->header_table.entries);
                free(cond_tree);
                free(new_prefix);
            } else {
                if (!is_subsumed(registry, new_prefix, prefix_len + y_count + 1, new_supp)) {
                    add_to_registry(registry, new_prefix, prefix_len + y_count + 1, new_supp);
                } else {
                    free(new_prefix);
                }
            }
            
            alloc_free(&local_alloc);
            
            for (size_t t = 0; t < cond_db.count; t++) {
                free(cond_db.array[t].items);
            }
        } else {
            if (!is_subsumed(registry, new_prefix, prefix_len + y_count + 1, new_supp)) {
                add_to_registry(registry, new_prefix, prefix_len + y_count + 1, new_supp);
            } else {
                free(new_prefix);
            }
        }
        free(cond_db.array);
    }
    
    free(item_to_supp);
}

/* --- DFS MINING: SPARSE (TOP-DOWN PSEUDO PROJECTION) --- */

typedef struct {
    uint32_t item;
    uint32_t support;
    FP_Node **nodes;
    size_t num_nodes;
    size_t cap_nodes;
} TDPP_Entry;

typedef struct {
    TDPP_Entry *entries;
    size_t count;
} TDPP_Table;

static void dfs_count(FP_Node *node, uint32_t *counts) {
    FP_Node *child = node->first_child;
    while (child) {
        counts[child->item] += child->count;
        dfs_count(child, counts);
        child = child->next_sibling;
    }
}

static void tdpp_dfs_link(FP_Node *node, TDPP_Table *ht, uint32_t *item_map) {
    FP_Node *child = node->first_child;
    while (child) {
        uint32_t idx = item_map[child->item];
        if (idx != ROOT_ITEM) {
            TDPP_Entry *entry = &ht->entries[idx];
            if (entry->num_nodes >= entry->cap_nodes) {
                entry->cap_nodes = entry->cap_nodes == 0 ? 4 : entry->cap_nodes * 2;
                entry->nodes = (FP_Node **)realloc(entry->nodes, sizeof(FP_Node*) * entry->cap_nodes);
            }
            entry->nodes[entry->num_nodes++] = child;
        }
        tdpp_dfs_link(child, ht, item_map);
        child = child->next_sibling;
    }
}

static void mine_tree_top_down(TDPP_Table *local_ht, uint32_t min_sup, uint32_t *prefix, size_t prefix_len, uint32_t max_item_id, size_t *total_fci, size_t *total_footprint) {
    for (size_t i = 0; i < local_ht->count; i++) {
        TDPP_Entry *entry = &local_ht->entries[i];
        uint32_t item_i = entry->item;
        uint32_t supp_i = entry->support;

        uint32_t *child_counts = calloc(max_item_id + 1, sizeof(uint32_t));
        for (size_t k = 0; k < entry->num_nodes; k++) {
            dfs_count(entry->nodes[k], child_counts);
        }

        size_t y_count = 0;
        size_t next_level_count = 0;
        for (uint32_t y = 0; y <= max_item_id; y++) {
            if (child_counts[y] == supp_i) {
                y_count++;
            } else if (child_counts[y] >= min_sup) {
                next_level_count++;
            }
        }

        uint32_t *merged_prefix = malloc(sizeof(uint32_t) * (prefix_len + 1 + y_count));
        if (prefix_len > 0) memcpy(merged_prefix, prefix, sizeof(uint32_t) * prefix_len);
        size_t idx = prefix_len;
        merged_prefix[idx++] = item_i;
        for (uint32_t y = 0; y <= max_item_id; y++) {
            if (child_counts[y] == supp_i) {
                merged_prefix[idx++] = y;
            }
        }
        qsort(merged_prefix, prefix_len + 1 + y_count, sizeof(uint32_t), cmp_uint32);

        // Upward Checking
        bool is_closed = true;
        uint32_t *up_counts = calloc(max_item_id + 1, sizeof(uint32_t));
        for (size_t k = 0; k < entry->num_nodes; k++) {
            FP_Node *n = entry->nodes[k];
            uint32_t weight = n->count;
            FP_Node *p = n->parent;
            while (p && p->item != ROOT_ITEM) {
                up_counts[p->item] += weight;
                p = p->parent;
            }
        }
        for (uint32_t y = 0; y <= max_item_id; y++) {
            if (up_counts[y] == supp_i && !in_array(y, merged_prefix, prefix_len + 1 + y_count)) {
                is_closed = false;
                break;
            }
        }
        free(up_counts);

        if (is_closed) {
            (*total_fci)++;
            (*total_footprint) += (prefix_len + 1 + y_count);

            if (next_level_count > 0) {
                TDPP_Table next_ht;
                next_ht.count = next_level_count;
                next_ht.entries = (TDPP_Entry *)calloc(next_level_count, sizeof(TDPP_Entry));
                
                uint32_t *item_map = (uint32_t *)malloc(sizeof(uint32_t) * (max_item_id + 1));
                for (uint32_t y = 0; y <= max_item_id; y++) item_map[y] = ROOT_ITEM;
                
                // Sort by support descending
                Header_Entry *temp_arr = malloc(sizeof(Header_Entry) * next_level_count);
                size_t t_idx = 0;
                for (uint32_t y = 0; y <= max_item_id; y++) {
                    if (child_counts[y] >= min_sup && child_counts[y] < supp_i) {
                        temp_arr[t_idx].item = y;
                        temp_arr[t_idx].support = child_counts[y];
                        t_idx++;
                    }
                }
                qsort(temp_arr, next_level_count, sizeof(Header_Entry), cmp_support_desc);

                for (size_t y = 0; y < next_level_count; y++) {
                    next_ht.entries[y].item = temp_arr[y].item;
                    next_ht.entries[y].support = temp_arr[y].support;
                    item_map[temp_arr[y].item] = (uint32_t)y;
                }
                free(temp_arr);

                for (size_t k = 0; k < entry->num_nodes; k++) {
                    tdpp_dfs_link(entry->nodes[k], &next_ht, item_map);
                }
                free(item_map);

                mine_tree_top_down(&next_ht, min_sup, merged_prefix, prefix_len + 1 + y_count, max_item_id, total_fci, total_footprint);

                for (size_t y = 0; y < next_level_count; y++) {
                    if (next_ht.entries[y].nodes) free(next_ht.entries[y].nodes);
                }
                free(next_ht.entries);
            }
        }
        
        free(merged_prefix);
        free(child_counts);
    }
}

/* --- MAIN ENTRY --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_CLOSETPLUS_Params *p = (DM_CLOSETPLUS_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[CLOSET+] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    Weighted_Trans_List db;
    db.count = ds->count;
    db.capacity = ds->count;
    db.max_id = ds->max_id;
    db.array = (Weighted_Trans *)malloc(sizeof(Weighted_Trans) * ds->count);
    
    for (size_t i = 0; i < ds->count; i++) {
        db.array[i].items = data[i].items;
        db.array[i].count = data[i].count;
        db.array[i].weight = 1;
    }

    FP_Allocator global_alloc;
    alloc_init(&global_alloc);

    FP_Tree *tree = build_tree(&db, min_sup, &global_alloc);
    free(db.array);

    size_t total_fci = 0;
    size_t total_footprint = 0;

    if (tree) {
        double avg_count = (double)tree->total_counts / (double)tree->total_nodes;
        bool is_dense = (avg_count >= 2.0);

        if (is_dense) {
            printf("[CLOSET+] Detected Dense Dataset (avg count: %.2f). Using Bottom-up Physical Projection.\n", avg_count);
            FCI_Registry registry;
            registry.max_support = (uint32_t)ds->count;
            registry.total_fci = 0;
            registry.total_footprint = 0;
            registry.support_map = (Closet_Itemset_List *)calloc(registry.max_support + 1, sizeof(Closet_Itemset_List));

            mine_tree_bottom_up(tree, min_sup, NULL, 0, (uint32_t)ds->count, &registry, &global_alloc);

            total_fci = registry.total_fci;
            total_footprint = registry.total_footprint;

            for (uint32_t i = 0; i <= registry.max_support; i++) {
                if (registry.support_map[i].capacity > 0) {
                    for (size_t j = 0; j < registry.support_map[i].count; j++) free(registry.support_map[i].array[j].items);
                    free(registry.support_map[i].array);
                }
            }
            free(registry.support_map);
        } else {
            printf("[CLOSET+] Detected Sparse Dataset (avg count: %.2f). Using Top-down Pseudo Projection.\n", avg_count);
            
            uint32_t *root_counts = calloc(tree->max_item_id + 1, sizeof(uint32_t));
            dfs_count(tree->root, root_counts);
            
            size_t y_count = 0;
            size_t next_level_count = 0;
            for (uint32_t y = 0; y <= tree->max_item_id; y++) {
                if (root_counts[y] == ds->count) y_count++;
                else if (root_counts[y] >= min_sup) next_level_count++;
            }

            uint32_t *merged_prefix = NULL;
            if (y_count > 0) {
                merged_prefix = malloc(sizeof(uint32_t) * y_count);
                size_t idx = 0;
                for (uint32_t y = 0; y <= tree->max_item_id; y++) {
                    if (root_counts[y] == ds->count) merged_prefix[idx++] = y;
                }
                qsort(merged_prefix, y_count, sizeof(uint32_t), cmp_uint32);
                
                total_fci++;
                total_footprint += y_count;
            }

            if (next_level_count > 0) {
                TDPP_Table root_ht;
                root_ht.count = next_level_count;
                root_ht.entries = (TDPP_Entry *)calloc(next_level_count, sizeof(TDPP_Entry));
                
                uint32_t *item_map = (uint32_t *)malloc(sizeof(uint32_t) * (tree->max_item_id + 1));
                for (uint32_t y = 0; y <= tree->max_item_id; y++) item_map[y] = ROOT_ITEM;

                Header_Entry *temp_arr = malloc(sizeof(Header_Entry) * next_level_count);
                size_t t_idx = 0;
                for (uint32_t y = 0; y <= tree->max_item_id; y++) {
                    if (root_counts[y] >= min_sup && root_counts[y] < ds->count) {
                        temp_arr[t_idx].item = y;
                        temp_arr[t_idx].support = root_counts[y];
                        t_idx++;
                    }
                }
                qsort(temp_arr, next_level_count, sizeof(Header_Entry), cmp_support_desc);

                for (size_t y = 0; y < next_level_count; y++) {
                    root_ht.entries[y].item = temp_arr[y].item;
                    root_ht.entries[y].support = temp_arr[y].support;
                    item_map[temp_arr[y].item] = (uint32_t)y;
                }
                free(temp_arr);

                tdpp_dfs_link(tree->root, &root_ht, item_map);
                free(item_map);

                mine_tree_top_down(&root_ht, min_sup, merged_prefix, y_count, tree->max_item_id, &total_fci, &total_footprint);

                for (size_t y = 0; y < next_level_count; y++) {
                    if (root_ht.entries[y].nodes) free(root_ht.entries[y].nodes);
                }
                free(root_ht.entries);
            }
            
            if (merged_prefix) free(merged_prefix);
            free(root_counts);
        }

        free(tree->header_table.entries);
        free(tree);
    }

    alloc_free(&global_alloc);

    printf("[CLOSET+] Complete. Total frequent closed itemsets found: %zu\n", total_fci);
    dm_bench_record_results(total_fci, total_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "closetplus",
    .name = "CLOSET+ Algorithm",
    .description = "Searching for the Best Strategies for Mining Frequent Closed Itemsets (Wang et al. 2003).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
